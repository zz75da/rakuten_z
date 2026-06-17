"""
One-shot MiniLM text feature encoder.

Reads X_train_update.csv, encodes the 'description' column using
paraphrase-multilingual-MiniLM-L12-v2, and saves the embeddings to
data/feature_cache/text_features_minilm.npy.

Run via:
    docker-compose --profile minilm run --rm minilm-encoder
"""
# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : script CLI autonome (legacy, hors API FastAPI) qui
# encode la colonne "description" avec
# paraphrase-multilingual-MiniLM-L12-v2 et sauvegarde le .npy
# resultant. Conserve comme alternative en ligne de commande a
# minilm-encoder/app.py (qui encode designation+description et
# gere aussi mpnet via /encode).
#
# Fonctions principales :
#   - log_memory(prefix="") : logge le pourcentage RAM utilisee
#     et la RAM disponible (psutil), pour suivre les pics memoire
#   - main() : pipeline complet - si OUTPUT_PATH existe deja,
#     ne refait rien (cache) ; sinon charge le CSV, encode
#     "description" par batches avec SentenceTransformer (CPU,
#     normalize_embeddings=False), sauvegarde le .npy (float32)
#
# Variables / constantes importantes :
#   - MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
#   - CSV_PATH (env TRAIN_CSV_X_PATH) : CSV source
#   - OUTPUT_PATH (env MINILM_CACHE_PATH) : .npy de sortie (384-d)
#   - BATCH_SIZE (env MINILM_BATCH_SIZE, def. 256)
#
# Dependances externes : numpy, pandas, psutil, tqdm,
# sentence-transformers
# ============================================================
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
