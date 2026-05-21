"""
CLIP ViT-B/32 text encoder service — lightweight FastAPI wrapper.

Uses transformers CLIPTokenizer + CLIPTextModel directly (text-only, ~170 MB)
instead of the full sentence-transformers CLIP wrapper which pulls the image
processor and breaks on newer transformers versions.

Produces 512-d text embeddings compatible with CLIP's embedding space.

Idle memory: ~50 MB (no model loaded).
During encoding: ~1.5 GB peak, then model is explicitly unloaded via gc.collect().

Endpoints:
  GET  /health   — liveness probe
  POST /encode   — start encoding (idempotent: skips if cache exists)
  GET  /status   — poll encoding progress
"""
import gc
import logging
import os
import threading

import numpy as np
import pandas as pd
from fastapi import FastAPI
from prometheus_client import make_asgi_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="CLIP Encoder", version="1.0")

HF_MODEL     = "openai/clip-vit-base-patch32"
CSV_PATH     = os.getenv("TRAIN_CSV_X_PATH",  "/app/data/X_train_update.csv")
OUTPUT_PATH  = os.getenv("CLIP_CACHE_PATH",   "/app/data/feature_cache/text_features_clip.npy")
BATCH_SIZE   = int(os.getenv("CLIP_BATCH_SIZE", "64"))


def _load_params() -> dict:
    for path in ["/app/params.yaml", "params.yaml"]:
        if os.path.exists(path):
            try:
                import yaml
                with open(path) as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


_PARAMS = _load_params()
_clip_cfg = _PARAMS.get("clip", {})
NORMALIZE_EMBEDDINGS = bool(_clip_cfg.get("normalize_embeddings", True))
_batch_override = _clip_cfg.get("batch_size")
if _batch_override:
    BATCH_SIZE = int(_batch_override)

logging.info(f"CLIP encoder: model={HF_MODEL} batch_size={BATCH_SIZE} normalize={NORMALIZE_EMBEDDINGS}")

_lock  = threading.Lock()
_state = {"status": "idle", "message": "Ready"}


def _encode_texts(texts, tokenizer, model, batch_size, normalize, state_ref, lock_ref):
    """Encode texts in batches, logging progress every 10% as clean log lines."""
    import torch

    n        = len(texts)
    n_batches = (n + batch_size - 1) // batch_size
    log_every = max(1, n_batches // 10)   # log ~10 times total

    all_embeddings = []
    for batch_idx, i in enumerate(range(0, n, batch_size)):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.pooler_output  # (batch, 512)
        if normalize:
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu().numpy())

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
            done = min(i + batch_size, n)
            pct  = done / n * 100
            msg  = f"Encoding {done}/{n} texts ({pct:.0f}%) — batch {batch_idx+1}/{n_batches}"
            logging.info(msg)
            with lock_ref:
                state_ref["message"] = msg

    return np.vstack(all_embeddings).astype(np.float32)


def _encode_worker():
    try:
        with _lock:
            _state.update({"status": "encoding", "message": "Loading dataset..."})

        df    = pd.read_csv(CSV_PATH)
        texts = df["description"].fillna("").tolist()
        n     = len(texts)
        logging.info(f"Dataset loaded: {n} samples")

        with _lock:
            _state["message"] = f"Loading CLIP text encoder from {HF_MODEL}..."

        from transformers import CLIPTokenizer, CLIPTextModel
        tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL)
        model     = CLIPTextModel.from_pretrained(HF_MODEL)
        model.eval()
        logging.info("CLIP text encoder loaded — encoding...")

        with _lock:
            _state["message"] = f"Encoding {n} texts (batch={BATCH_SIZE}, normalize={NORMALIZE_EMBEDDINGS})..."

        embeddings = _encode_texts(texts, tokenizer, model, BATCH_SIZE, NORMALIZE_EMBEDDINGS, _state, _lock)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        np.save(OUTPUT_PATH, embeddings)
        size_mb = os.path.getsize(OUTPUT_PATH) / 1024 ** 2
        logging.info(f"Saved {size_mb:.0f} MB → {OUTPUT_PATH}  shape={embeddings.shape}")

        del tokenizer, model, embeddings, df, texts
        gc.collect()
        logging.info("CLIP model unloaded — RAM freed for training")

        with _lock:
            _state.update({
                "status": "done",
                "message": f"Saved {size_mb:.0f} MB ({n} × 512-d). Model unloaded.",
            })

    except Exception as exc:
        logging.exception("CLIP encoding failed")
        with _lock:
            _state.update({"status": "error", "message": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/encode")
def encode():
    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / 1024 ** 2
        logging.info(f"Cache hit — {size_mb:.0f} MB at {OUTPUT_PATH}, skipping encoding")
        with _lock:
            _state.update({"status": "done", "message": f"Cache exists ({size_mb:.0f} MB)"})
        return {"status": "done", "message": "Cache already exists — skipping"}

    with _lock:
        current = _state["status"]

    if current == "encoding":
        return {"status": "encoding", "message": "Already running"}

    threading.Thread(target=_encode_worker, daemon=True).start()
    return {"status": "encoding", "message": "Encoding started"}


@app.get("/status")
def status():
    if os.path.exists(OUTPUT_PATH) and _state["status"] != "error":
        with _lock:
            _state.update({"status": "done"})
    with _lock:
        return dict(_state)


app.mount("/metrics", make_asgi_app())
