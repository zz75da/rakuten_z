"""
CLIP ViT-B/32 text encoder service — lightweight FastAPI wrapper.

Uses transformers CLIPTokenizer + CLIPTextModel directly (text-only, ~170 MB)
instead of the full sentence-transformers CLIP wrapper which pulls the image
processor and breaks on newer transformers versions.

Produces 512-d text embeddings compatible with CLIP's embedding space.

Idle memory: ~50 MB (no model loaded).
During encoding: ~1.5 GB peak, then model is explicitly unloaded via gc.collect().

Endpoints:
  GET    /health         — liveness probe (exposes current config)
  POST   /encode         — start encoding (idempotent, validates cache params)
  GET    /status         — poll encoding progress
  DELETE /cache          — invalidate cache so next /encode re-encodes from scratch

Error-proofing applied (lessons from MiniLM incidents):
  - Atomic write: .npy saved to tmp first, then os.replace() — no partial files
  - Param metadata sidecar (.meta): cache hit rejected if normalize/model changed
  - Race-condition-safe /encode: check+set status in single lock
  - Pre-flight CSV check before spawning thread
  - try/finally model cleanup — RAM freed even on error
  - Minimum file size check in /status — empty/truncated file ≠ valid
  - HF model persisted via volume mount — no re-download on container rebuild
"""
import gc
import json
import logging
import os
import threading
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="CLIP Encoder", version="1.0")

HF_MODEL    = "openai/clip-vit-base-patch32"
CSV_PATH    = os.getenv("TRAIN_CSV_X_PATH",  "/app/data/X_train_update.csv")
OUTPUT_PATH = os.getenv("CLIP_CACHE_PATH",   "/app/data/feature_cache/text_features_clip.npy")
META_PATH   = OUTPUT_PATH + ".meta"
BATCH_SIZE  = int(os.getenv("CLIP_BATCH_SIZE", "64"))

# Minimum expected file size: 1000 samples × 512 dims × 4 bytes — anything
# smaller is a truncated/corrupt write and must not be used as a valid cache.
_MIN_CACHE_BYTES = 1000 * 512 * 4


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

logging.info(
    f"CLIP encoder ready: model={HF_MODEL} "
    f"batch_size={BATCH_SIZE} normalize={NORMALIZE_EMBEDDINGS}"
)

_lock  = threading.Lock()
_state = {"status": "idle", "message": "Ready"}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_is_valid() -> bool:
    """
    Returns True only when the cache file:
      1. Exists and is large enough to be a real encoding (not a truncated write)
      2. Has a sidecar .meta file with matching model and normalize_embeddings param

    This prevents the two main historical failure modes:
      - Partial file from a mid-write crash (atomic write prevents this now,
        but size check is a belt-and-suspenders defence)
      - Stale cache from a previous run with different normalize_embeddings
        (what broke MiniLM when normalize was toggled)
    """
    if not os.path.exists(OUTPUT_PATH):
        return False
    if os.path.getsize(OUTPUT_PATH) < _MIN_CACHE_BYTES:
        logging.warning(
            f"Cache file too small ({os.path.getsize(OUTPUT_PATH)} B < "
            f"{_MIN_CACHE_BYTES} B) — treating as corrupt"
        )
        return False
    if not os.path.exists(META_PATH):
        logging.warning("Cache file exists but .meta is missing — treating as stale")
        return False
    try:
        with open(META_PATH) as f:
            meta = json.load(f)
        if meta.get("model") != HF_MODEL:
            logging.warning(
                f"Cache model mismatch: stored={meta.get('model')} current={HF_MODEL}"
            )
            return False
        if meta.get("normalize") != NORMALIZE_EMBEDDINGS:
            logging.warning(
                f"Cache normalize mismatch: stored={meta.get('normalize')} "
                f"current={NORMALIZE_EMBEDDINGS} — re-encoding required"
            )
            return False
        return True
    except Exception as exc:
        logging.warning(f"Could not read .meta file: {exc} — treating cache as stale")
        return False


def _write_cache_meta(n_samples: int, shape: tuple):
    meta = {
        "model":      HF_MODEL,
        "normalize":  NORMALIZE_EMBEDDINGS,
        "n_samples":  n_samples,
        "shape":      list(shape),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_texts(texts, tokenizer, model, batch_size, normalize, state_ref, lock_ref):
    """Encode texts in batches, logging progress every ~10% as clean INFO lines."""
    import torch

    n         = len(texts)
    n_batches = (n + batch_size - 1) // batch_size
    log_every = max(1, n_batches // 10)

    all_embeddings = []
    for batch_idx, i in enumerate(range(0, n, batch_size)):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.pooler_output           # (batch, 512)
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
    tokenizer = None
    model     = None
    t_start   = time.time()

    try:
        with _lock:
            _state.update({"status": "encoding", "message": "Loading dataset..."})

        # --- Pre-flight: CSV must exist ---
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(
                f"Training CSV not found at {CSV_PATH}. "
                "Check TRAIN_CSV_X_PATH and volume mounts."
            )

        df    = pd.read_csv(CSV_PATH)
        texts = df["description"].fillna("").tolist()
        n     = len(texts)
        logging.info(f"Dataset loaded: {n} samples")

        with _lock:
            _state["message"] = f"Loading CLIP text encoder ({HF_MODEL})..."

        from transformers import CLIPTokenizer, CLIPTextModel
        tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL)
        model     = CLIPTextModel.from_pretrained(HF_MODEL)
        model.eval()
        logging.info("CLIP text encoder loaded — encoding...")

        with _lock:
            _state["message"] = (
                f"Encoding {n} texts "
                f"(batch={BATCH_SIZE}, normalize={NORMALIZE_EMBEDDINGS})..."
            )

        embeddings = _encode_texts(
            texts, tokenizer, model, BATCH_SIZE, NORMALIZE_EMBEDDINGS, _state, _lock
        )

        # --- Atomic write: write to .tmp then rename so a crash mid-write
        #     never leaves a partial file that looks like a valid cache. ---
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        tmp_path = OUTPUT_PATH + ".tmp"
        np.save(tmp_path, embeddings)
        os.replace(tmp_path, OUTPUT_PATH)         # atomic on Linux/ext4

        _write_cache_meta(n, embeddings.shape)

        elapsed = time.time() - t_start
        size_mb = os.path.getsize(OUTPUT_PATH) / 1024 ** 2
        logging.info(
            f"Saved {size_mb:.0f} MB → {OUTPUT_PATH}  "
            f"shape={embeddings.shape}  elapsed={elapsed/60:.1f} min"
        )

        done_msg = (
            f"Saved {size_mb:.0f} MB ({n} × 512-d) in {elapsed/60:.1f} min. "
            "Model unloaded."
        )
        with _lock:
            _state.update({"status": "done", "message": done_msg})

    except Exception as exc:
        logging.exception("CLIP encoding failed")
        # Remove any partial tmp file so a retry starts clean
        tmp_path = OUTPUT_PATH + ".tmp"
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        with _lock:
            _state.update({"status": "error", "message": str(exc)})

    finally:
        # Always free memory — even if encoding failed partway through
        del tokenizer, model
        gc.collect()
        logging.info("CLIP model unloaded — RAM freed")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe — also exposes current config for debugging."""
    cache_ok = _cache_is_valid()
    cache_mb = (
        os.path.getsize(OUTPUT_PATH) / 1024 ** 2
        if os.path.exists(OUTPUT_PATH) else 0
    )
    return {
        "status": "ok",
        "model": HF_MODEL,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "batch_size": BATCH_SIZE,
        "cache_valid": cache_ok,
        "cache_mb": round(cache_mb, 1),
    }


@app.post("/encode")
def encode():
    """
    Start encoding (idempotent).

    Cache hit: only accepted if the .meta file confirms current model and
    normalize_embeddings match — prevents silently using stale embeddings
    after a param change (the MiniLM incident).

    Race-condition-safe: status check and update happen inside the same lock
    so two concurrent requests cannot both start encoding threads.
    """
    # Pre-flight: fail fast if the CSV doesn't exist
    if not os.path.exists(CSV_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Training CSV not found at {CSV_PATH}. Check volume mounts.",
        )

    with _lock:
        # Valid cache → return immediately without re-encoding
        if _cache_is_valid():
            size_mb = os.path.getsize(OUTPUT_PATH) / 1024 ** 2
            _state.update({"status": "done", "message": f"Cache valid ({size_mb:.0f} MB)"})
            logging.info(f"Cache hit (valid) — {size_mb:.0f} MB, skipping encoding")
            return {"status": "done", "message": "Cache already exists and params match"}

        # Guard against concurrent requests
        if _state["status"] == "encoding":
            return {"status": "encoding", "message": "Already running"}

        # Clean up any leftover error state before retry
        if _state["status"] == "error":
            logging.info("Clearing previous error state before retry")

        # Mark as encoding inside the lock so a concurrent request sees it
        _state.update({"status": "encoding", "message": "Starting..."})

    threading.Thread(target=_encode_worker, daemon=True).start()
    return {"status": "encoding", "message": "Encoding started"}


@app.get("/status")
def status():
    """
    Poll encoding progress.
    Returns "done" only when cache file is valid (exists + correct size + params match).
    """
    with _lock:
        current = dict(_state)

    if current["status"] != "error":
        if _cache_is_valid():
            with _lock:
                _state.update({"status": "done"})
            current["status"] = "done"
        elif current["status"] == "done":
            # State says done but cache is invalid — encoding was interrupted
            with _lock:
                _state.update({
                    "status": "error",
                    "message": "Cache file missing or invalid after encoding reported done",
                })
            current = dict(_state)

    return current


@app.delete("/cache")
def delete_cache():
    """
    Invalidate the feature cache so the next POST /encode re-encodes from scratch.
    Useful after changing params (e.g. normalize_embeddings) without restarting
    the container — avoids the need to docker exec + rm manually.
    """
    with _lock:
        if _state["status"] == "encoding":
            raise HTTPException(
                status_code=409,
                detail="Encoding is in progress — wait for it to finish before deleting cache",
            )

    deleted = []
    for path in [OUTPUT_PATH, META_PATH, OUTPUT_PATH + ".tmp"]:
        if os.path.exists(path):
            os.unlink(path)
            deleted.append(os.path.basename(path))
            logging.info(f"Deleted: {path}")

    with _lock:
        _state.update({"status": "idle", "message": "Cache deleted — ready to re-encode"})

    return {"deleted": deleted, "message": "Cache cleared. POST /encode to re-encode."}


app.mount("/metrics", make_asgi_app())
