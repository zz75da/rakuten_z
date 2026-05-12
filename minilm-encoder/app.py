"""
MiniLM encoder service — lightweight FastAPI wrapper around sentence-transformers.

Idle memory: ~50 MB (no model loaded).
During encoding: ~2 GB peak, then model is explicitly unloaded via gc.collect()
so train-api has full RAM available when MiniLM training starts.

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="MiniLM Encoder", version="1.0")

MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"
CSV_PATH    = os.getenv("TRAIN_CSV_X_PATH",  "/app/data/X_train_update.csv")
OUTPUT_PATH = os.getenv("MINILM_CACHE_PATH", "/app/data/feature_cache/text_features_minilm.npy")
BATCH_SIZE  = int(os.getenv("MINILM_BATCH_SIZE", "256"))

_lock  = threading.Lock()
_state = {"status": "idle", "message": "Ready"}


def _encode_worker():
    try:
        with _lock:
            _state.update({"status": "encoding", "message": "Loading dataset..."})

        df    = pd.read_csv(CSV_PATH)
        texts = df["description"].fillna("").tolist()
        n     = len(texts)
        logging.info(f"Dataset loaded: {n} samples")

        with _lock:
            _state["message"] = f"Loading SentenceTransformer ({n} samples queued)..."

        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(MODEL_NAME, device="cpu")
        logging.info("Model loaded — encoding...")

        with _lock:
            _state["message"] = f"Encoding {n} texts (batch={BATCH_SIZE})..."

        embeddings = encoder.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        np.save(OUTPUT_PATH, embeddings.astype(np.float32))
        size_mb = os.path.getsize(OUTPUT_PATH) / 1024 ** 2
        logging.info(f"Saved {size_mb:.0f} MB → {OUTPUT_PATH}")

        # Unload model immediately so train-api has full RAM for MiniLM training
        del encoder, embeddings, df, texts
        gc.collect()
        logging.info("Model unloaded — RAM freed for training")

        with _lock:
            _state.update({
                "status": "done",
                "message": f"Saved {size_mb:.0f} MB. Model unloaded, RAM free for training.",
            })

    except Exception as exc:
        logging.exception("Encoding failed")
        with _lock:
            _state.update({"status": "error", "message": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/encode")
def encode():
    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / 1024 ** 2
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
