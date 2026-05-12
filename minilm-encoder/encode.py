"""
One-shot MiniLM text feature encoder.

Reads X_train_update.csv, encodes the 'description' column using
paraphrase-multilingual-MiniLM-L12-v2, and saves the embeddings to
data/feature_cache/text_features_minilm.npy.

Run via:
    docker-compose --profile minilm run --rm minilm-encoder
"""
import os
import numpy as np
import pandas as pd
import psutil
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"
CSV_PATH     = os.getenv("TRAIN_CSV_X_PATH", "/app/data/X_train_update.csv")
OUTPUT_PATH  = os.getenv("MINILM_CACHE_PATH", "/app/data/feature_cache/text_features_minilm.npy")
BATCH_SIZE   = int(os.getenv("MINILM_BATCH_SIZE", "256"))


def log_memory(prefix=""):
    mem = psutil.virtual_memory()
    logging.info(f"{prefix} RAM: {mem.percent:.1f}% used, {mem.available / 1024**3:.2f} GB free")


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / 1024**2
        logging.info(f"Cache already exists ({size_mb:.0f} MB): {OUTPUT_PATH} — skipping.")
        return

    logging.info(f"Loading dataset from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    texts = df["description"].fillna("").tolist()
    logging.info(f"Loaded {len(texts)} samples")
    log_memory("Before encoding")

    logging.info(f"Loading SentenceTransformer: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME, device="cpu")
    log_memory("After model load")

    logging.info(f"Encoding {len(texts)} texts (batch_size={BATCH_SIZE})...")
    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    log_memory("After encoding")

    logging.info(f"Embeddings shape: {embeddings.shape}  dtype: {embeddings.dtype}")
    np.save(OUTPUT_PATH, embeddings.astype(np.float32))
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024**2
    logging.info(f"Saved to {OUTPUT_PATH} ({size_mb:.0f} MB)")
    log_memory("Done")


if __name__ == "__main__":
    main()
