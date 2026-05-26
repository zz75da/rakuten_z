"""
MiniLM encoder service — lightweight FastAPI wrapper around sentence-transformers.

Model: intfloat/multilingual-e5-small (overridable via MINILM_MODEL env var)
  - 384-d output, 100 languages, trained on retrieval+classification benchmarks
  - Uses "query: " prefix per E5 specification for optimal classification performance
  - Encodes designation + cleaned description (designation always present; description
    HTML-unescaped and tag-stripped, concatenated when available)
  - max_seq_length=128: covers full designation + start of description; 16× less
    attention memory than the E5-small default of 512 (quadratic scaling)

Idle memory: ~50 MB (no model loaded).
During encoding: ~2 GB peak, then model is explicitly unloaded via gc.collect()
so train-api has full RAM available when MiniLM training starts.

Endpoints:
  GET  /health   — liveness probe (exposes current config + cache validity)
  POST /encode   — start encoding (idempotent, validates model+params match)
  GET  /status   — poll encoding progress
"""
import gc
import json
import logging
import os
import threading
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI
from prometheus_client import make_asgi_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="MiniLM Encoder", version="2.0")

MODEL_NAME  = os.getenv("MINILM_MODEL", "intfloat/multilingual-e5-small")
CSV_PATH    = os.getenv("TRAIN_CSV_X_PATH",  "/app/data/X_train_update.csv")
OUTPUT_PATH = os.getenv("MINILM_CACHE_PATH", "/app/data/feature_cache/text_features_minilm.npy")
META_PATH   = OUTPUT_PATH + ".meta"
BATCH_SIZE  = int(os.getenv("MINILM_BATCH_SIZE", "256"))

# E5 models require "query: " prefix for retrieval/classification tasks
_E5_PREFIX = "query: " if "e5" in MODEL_NAME.lower() else ""

_MIN_CACHE_BYTES = 1000 * 384 * 4  # 1000 samples × 384 dims × 4 bytes

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
NORMALIZE_EMBEDDINGS = bool(_PARAMS.get("minilm", {}).get("normalize_embeddings", True))

logging.info(
    f"MiniLM encoder ready: model={MODEL_NAME} "
    f"batch_size={BATCH_SIZE} normalize={NORMALIZE_EMBEDDINGS} "
    f"prefix={'yes' if _E5_PREFIX else 'no'}"
)

_lock  = threading.Lock()
_state = {"status": "idle", "message": "Ready"}


def _cache_is_valid() -> bool:
    try:
        if not os.path.exists(OUTPUT_PATH):
            return False
        if os.path.getsize(OUTPUT_PATH) < _MIN_CACHE_BYTES:
            logging.warning("Cache file too small — treating as corrupt")
            return False
        if not os.path.exists(META_PATH):
            logging.warning("Cache .meta missing — treating as stale")
            return False
        with open(META_PATH) as f:
            meta = json.load(f)
        if meta.get("model") != MODEL_NAME:
            logging.warning(f"Cache model mismatch: stored={meta.get('model')} current={MODEL_NAME}")
            return False
        if meta.get("normalize") != NORMALIZE_EMBEDDINGS:
            logging.warning(f"Cache normalize mismatch: stored={meta.get('normalize')} current={NORMALIZE_EMBEDDINGS}")
            return False
        return True
    except Exception as exc:
        logging.warning(f"Could not validate cache: {exc}")
        return False


def _write_cache_meta(n_samples: int, shape: tuple):
    meta = {
        "model":      MODEL_NAME,
        "normalize":  NORMALIZE_EMBEDDINGS,
        "n_samples":  n_samples,
        "shape":      list(shape),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def _encode_worker():
    try:
        with _lock:
            _state.update({"status": "encoding", "message": "Loading dataset..."})

        import html as _html, re as _re

        df = pd.read_csv(CSV_PATH)

        def _clean(text):
            if not isinstance(text, str) or not text.strip():
                return ""
            text = _html.unescape(text)              # &eacute; → é
            text = _re.sub(r"<[^>]+>", " ", text)   # strip HTML tags
            return " ".join(text.split())            # normalise whitespace

        # Concatenate designation (always present) + description (when available, cleaned)
        desig = df["designation"].fillna("").apply(_clean)
        descr = df["description"].fillna("").apply(_clean)
        raw   = (desig + " " + descr).str.strip().tolist()
        texts = [(_E5_PREFIX + t) if t else _E5_PREFIX for t in raw]
        n = len(texts)
        logging.info(f"Dataset loaded: {n} samples (prefix='{_E5_PREFIX or 'none'}')")

        with _lock:
            _state["message"] = f"Loading {MODEL_NAME} ({n} samples queued)..."

        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(MODEL_NAME, device="cpu")
        encoder.max_seq_length = 128  # truncate to 128 tokens: covers full designation + start of
        # description; reduces attention memory 16× vs default 512 (128²→512² is quadratic)
        logging.info(
            f"Model loaded — encoding {n} texts in batches of {BATCH_SIZE} "
            f"(max_seq_length={encoder.max_seq_length})..."
        )

        # Encode in manual batches so we can log progress every ~10%
        n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
        log_every = max(1, n_batches // 10)
        all_embeddings = []

        for batch_idx, start in enumerate(range(0, n, BATCH_SIZE)):
            batch = texts[start : start + BATCH_SIZE]
            batch_emb = encoder.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
            )
            all_embeddings.append(batch_emb)

            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
                done = min(start + BATCH_SIZE, n)
                pct  = done / n * 100
                msg  = f"Encoding {done}/{n} texts ({pct:.0f}%) — batch {batch_idx+1}/{n_batches}"
                logging.info(msg)
                with _lock:
                    _state["message"] = msg

        import numpy as _np
        embeddings = _np.vstack(all_embeddings).astype(np.float32)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        tmp_path = OUTPUT_PATH + ".tmp"
        with open(tmp_path, "wb") as _f:
            np.save(_f, embeddings.astype(np.float32))
        os.replace(tmp_path, OUTPUT_PATH)
        _write_cache_meta(n, embeddings.shape)

        size_mb = os.path.getsize(OUTPUT_PATH) / 1024 ** 2
        logging.info(f"Saved {size_mb:.0f} MB → {OUTPUT_PATH}  shape={embeddings.shape}")

        del encoder, embeddings, df, texts
        gc.collect()
        logging.info("Model unloaded — RAM freed for training")

        with _lock:
            _state.update({
                "status": "done",
                "message": f"Saved {size_mb:.0f} MB ({n} × 384-d). Model unloaded.",
            })

    except Exception as exc:
        logging.exception("Encoding failed")
        with _lock:
            _state.update({"status": "error", "message": str(exc)})


@app.get("/health")
def health():
    cache_ok = _cache_is_valid()
    cache_mb = round(os.path.getsize(OUTPUT_PATH) / 1024**2, 1) if os.path.exists(OUTPUT_PATH) else 0
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "e5_prefix": bool(_E5_PREFIX),
        "batch_size": BATCH_SIZE,
        "cache_valid": cache_ok,
        "cache_mb": cache_mb,
    }


@app.post("/encode")
def encode():
    if _cache_is_valid():
        size_mb = round(os.path.getsize(OUTPUT_PATH) / 1024**2, 1)
        logging.info(f"Cache valid ({size_mb} MB) — skipping encoding")
        with _lock:
            _state.update({"status": "done", "message": f"Cache already exists and params match"})
        return {"status": "done", "message": "Cache already exists and params match"}

    with _lock:
        current = _state["status"]
    if current == "encoding":
        return {"status": "encoding", "message": "Already running"}

    threading.Thread(target=_encode_worker, daemon=True).start()
    return {"status": "encoding", "message": "Encoding started"}


@app.get("/status")
def status():
    if _cache_is_valid() and _state["status"] != "error":
        with _lock:
            _state.update({"status": "done"})
    with _lock:
        return dict(_state)


app.mount("/metrics", make_asgi_app())
