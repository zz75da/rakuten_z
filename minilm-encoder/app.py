"""
Dual-encoder service — encodes the dataset with TWO sentence-transformer models
sequentially, freeing RAM between them.

  Model 1 (MiniLM): paraphrase-multilingual-MiniLM-L12-v2  → text_features_minilm.npy (384-d)
  Model 2 (mpnet):  paraphrase-multilingual-mpnet-base-v2   → text_features_mpnet.npy  (768-d)

Both outputs are produced in a single /encode call.  Models are loaded and unloaded
one at a time so peak RAM never exceeds ~2 GB.

Idle memory: ~50 MB (no model loaded).
During encoding: ~2 GB peak (one model at a time), then model is explicitly unloaded
via gc.collect() so train-api has full RAM when training starts.

Endpoints:
  GET  /health   — liveness probe (exposes cache validity for both models)
  POST /encode   — start encoding (idempotent: skips models whose cache is already valid)
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

app = FastAPI(title="MiniLM+mpnet Encoder", version="3.0")

# ── Model configuration ──────────────────────────────────────────────────────
MINILM_MODEL  = os.getenv("MINILM_MODEL",  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MPNET_MODEL   = os.getenv("MPNET_MODEL",   "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

CSV_PATH      = os.getenv("TRAIN_CSV_X_PATH",  "/app/data/X_train_update.csv")
MINILM_OUTPUT = os.getenv("MINILM_CACHE_PATH", "/app/data/feature_cache/text_features_minilm.npy")
MPNET_OUTPUT  = os.getenv("MPNET_CACHE_PATH",  "/app/data/feature_cache/text_features_mpnet.npy")
BATCH_SIZE    = int(os.getenv("MINILM_BATCH_SIZE", "32"))

_MINILM_META = MINILM_OUTPUT + ".meta"
_MPNET_META  = MPNET_OUTPUT + ".meta"

# Minimum cache size sanity checks
_MIN_MINILM_BYTES = 1000 * 384 * 4   # 1000 samples × 384 dims × float32
_MIN_MPNET_BYTES  = 1000 * 768 * 4   # 1000 samples × 768 dims × float32


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
# Paraphrase models: normalize=False — raw embeddings preserve scale information
# that helps the dense classification head distinguish categories.
# (E5-small needed normalize=True because it uses cosine retrieval objective;
#  paraphrase models with NN head on top do not benefit from normalization.)
MINILM_NORMALIZE = bool(_PARAMS.get("minilm", {}).get("normalize_embeddings", False))
MPNET_NORMALIZE  = bool(_PARAMS.get("mpnet",  {}).get("normalize_embeddings", False))

logging.info(
    f"Dual encoder ready: "
    f"minilm={MINILM_MODEL} (normalize={MINILM_NORMALIZE}) | "
    f"mpnet={MPNET_MODEL} (normalize={MPNET_NORMALIZE}) | "
    f"batch_size={BATCH_SIZE}"
)

_lock  = threading.Lock()
_state = {
    "status":  "idle",
    "message": "Ready",
    "minilm":  "unknown",
    "mpnet":   "unknown",
}


# ── Cache validation ─────────────────────────────────────────────────────────

def _cache_valid(output_path: str, model_name: str, min_bytes: int, meta_path: str) -> bool:
    """Check that the .npy cache file exists, is non-trivial, and matches the model name."""
    try:
        if not os.path.exists(output_path):
            return False
        if os.path.getsize(output_path) < min_bytes:
            logging.warning(f"Cache too small for {model_name}: {output_path}")
            return False
        if not os.path.exists(meta_path):
            logging.warning(f"Cache .meta missing for {model_name}: {meta_path}")
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("model") != model_name:
            logging.warning(
                f"Cache model mismatch ({output_path}): "
                f"stored={meta.get('model')} current={model_name}"
            )
            return False
        return True
    except Exception as exc:
        logging.warning(f"Could not validate cache {output_path}: {exc}")
        return False


def _minilm_cache_valid() -> bool:
    return _cache_valid(MINILM_OUTPUT, MINILM_MODEL, _MIN_MINILM_BYTES, _MINILM_META)


def _mpnet_cache_valid() -> bool:
    return _cache_valid(MPNET_OUTPUT, MPNET_MODEL, _MIN_MPNET_BYTES, _MPNET_META)


def _all_caches_valid() -> bool:
    return _minilm_cache_valid() and _mpnet_cache_valid()


# ── Cache metadata helper ────────────────────────────────────────────────────

def _write_meta(output_path: str, meta_path: str, model_name: str,
                n_samples: int, shape: tuple):
    meta = {
        "model":      model_name,
        "n_samples":  n_samples,
        "shape":      list(shape),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ── Per-model encoding helper ─────────────────────────────────────────────────

def _encode_one_model(model_name: str, output_path: str, meta_path: str,
                      texts: list, normalize: bool, label: str) -> float:
    """
    Load one SentenceTransformer, encode all texts, write to disk, unload.
    Returns the output file size in MB.
    max_seq_length=128: covers full designation + start of description;
    16× less attention memory than default 512 (quadratic scaling).
    """
    n = len(texts)
    logging.info(f"[{label}] Loading {model_name}...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model_name, device="cpu")
    encoder.max_seq_length = 128
    logging.info(f"[{label}] Encoding {n} texts in batches of {BATCH_SIZE}...")

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
            normalize_embeddings=normalize,
        )
        all_embeddings.append(batch_emb)

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
            done = min(start + BATCH_SIZE, n)
            pct  = done / n * 100
            logging.info(
                f"[{label}] {done}/{n} ({pct:.0f}%) — batch {batch_idx+1}/{n_batches}"
            )
            with _lock:
                _state["message"] = f"[{label}] encoding {done}/{n} ({pct:.0f}%)"

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "wb") as _f:
        np.save(_f, embeddings)
    os.replace(tmp_path, output_path)
    _write_meta(output_path, meta_path, model_name, n, embeddings.shape)

    size_mb = os.path.getsize(output_path) / 1024 ** 2
    logging.info(
        f"[{label}] Saved {size_mb:.0f} MB → {output_path}  shape={embeddings.shape}"
    )

    del encoder, embeddings
    gc.collect()
    logging.info(f"[{label}] Model unloaded — RAM freed")
    return size_mb


# ── Background worker ─────────────────────────────────────────────────────────

def _encode_worker():
    try:
        with _lock:
            _state.update({
                "status":  "encoding",
                "message": "Loading dataset...",
                "minilm":  "pending",
                "mpnet":   "pending",
            })

        import html as _html, re as _re

        df = pd.read_csv(CSV_PATH)

        def _clean(text):
            if not isinstance(text, str) or not text.strip():
                return ""
            text = _html.unescape(text)              # &eacute; → é
            text = _re.sub(r"<[^>]+>", " ", text)   # strip HTML tags
            return " ".join(text.split())            # normalise whitespace

        desig = df["designation"].fillna("").apply(_clean)
        descr = df["description"].fillna("").apply(_clean)
        # No task prefix — paraphrase models don't use "query:/passage:" prefixes
        texts = (desig + " " + descr).str.strip().tolist()
        texts = [t if t else "" for t in texts]
        n = len(texts)
        logging.info(f"Dataset loaded: {n} samples (no prefix)")

        # ── 1. MiniLM — paraphrase-multilingual-MiniLM-L12-v2 (384-d) ────────
        if _minilm_cache_valid():
            size_mb = round(os.path.getsize(MINILM_OUTPUT) / 1024**2, 1)
            logging.info(f"[MiniLM] Cache valid ({size_mb} MB) — skipping")
            with _lock:
                _state["minilm"] = f"cached ({size_mb} MB)"
        else:
            with _lock:
                _state.update({
                    "minilm":  "encoding",
                    "message": f"[MiniLM] encoding {n} texts...",
                })
            size_mb = _encode_one_model(
                MINILM_MODEL, MINILM_OUTPUT, _MINILM_META,
                texts, MINILM_NORMALIZE, "MiniLM",
            )
            with _lock:
                _state["minilm"] = f"done ({size_mb:.0f} MB)"

        # ── 2. mpnet — paraphrase-multilingual-mpnet-base-v2 (768-d) ─────────
        if _mpnet_cache_valid():
            size_mb = round(os.path.getsize(MPNET_OUTPUT) / 1024**2, 1)
            logging.info(f"[mpnet] Cache valid ({size_mb} MB) — skipping")
            with _lock:
                _state["mpnet"] = f"cached ({size_mb} MB)"
        else:
            with _lock:
                _state.update({
                    "mpnet":   "encoding",
                    "message": f"[mpnet] encoding {n} texts...",
                })
            size_mb = _encode_one_model(
                MPNET_MODEL, MPNET_OUTPUT, _MPNET_META,
                texts, MPNET_NORMALIZE, "mpnet",
            )
            with _lock:
                _state["mpnet"] = f"done ({size_mb:.0f} MB)"

        del df, texts
        gc.collect()
        logging.info("All encoding complete — RAM freed for training")

        with _lock:
            _state.update({
                "status":  "done",
                "message": (
                    f"Both caches ready — "
                    f"minilm: {_state['minilm']} | mpnet: {_state['mpnet']}"
                ),
            })

    except Exception as exc:
        logging.exception("Encoding failed")
        with _lock:
            _state.update({"status": "error", "message": str(exc)})


# ── FastAPI endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    minilm_ok = _minilm_cache_valid()
    mpnet_ok  = _mpnet_cache_valid()
    minilm_mb = round(os.path.getsize(MINILM_OUTPUT) / 1024**2, 1) if os.path.exists(MINILM_OUTPUT) else 0
    mpnet_mb  = round(os.path.getsize(MPNET_OUTPUT)  / 1024**2, 1) if os.path.exists(MPNET_OUTPUT)  else 0
    return {
        "status":             "ok",
        "minilm_model":       MINILM_MODEL,
        "mpnet_model":        MPNET_MODEL,
        "batch_size":         BATCH_SIZE,
        "minilm_normalize":   MINILM_NORMALIZE,
        "mpnet_normalize":    MPNET_NORMALIZE,
        "minilm_cache_valid": minilm_ok,
        "mpnet_cache_valid":  mpnet_ok,
        "minilm_cache_mb":    minilm_mb,
        "mpnet_cache_mb":     mpnet_mb,
    }


@app.post("/encode")
def encode():
    if _all_caches_valid():
        minilm_mb = round(os.path.getsize(MINILM_OUTPUT) / 1024**2, 1)
        mpnet_mb  = round(os.path.getsize(MPNET_OUTPUT)  / 1024**2, 1)
        msg = f"Both caches valid — minilm: {minilm_mb} MB, mpnet: {mpnet_mb} MB"
        logging.info(msg)
        with _lock:
            _state.update({"status": "done", "message": msg})
        return {"status": "done", "message": msg}

    with _lock:
        current = _state["status"]
    if current == "encoding":
        return {"status": "encoding", "message": "Already running"}

    threading.Thread(target=_encode_worker, daemon=True).start()
    return {"status": "encoding", "message": "Encoding started (MiniLM then mpnet)"}


@app.get("/status")
def status():
    if _all_caches_valid() and _state["status"] != "error":
        with _lock:
            _state["status"] = "done"
    with _lock:
        return dict(_state)


app.mount("/metrics", make_asgi_app())
