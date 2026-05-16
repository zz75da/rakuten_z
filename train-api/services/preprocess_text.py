import os
import spacy
import numpy as np
import pandas as pd
import pickle
import logging
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import psutil

# === Constants ===
TEXT_FEATURES_LIMIT = 5000
TEXT_FEATURES_FILE = "data/feature_cache/text_features.npy"
TEXT_VECTORIZER_FILE = "data/text_vectorizer.pkl"
BATCH_SIZE = 5000  # adjust based on memory
# n_process=1 is faster than 6 inside Docker/WSL2: forking 6 worker processes
# per batch (17 batches × fork/join overhead) exceeds the gain from parallelism
# for this dataset size. Set via env var to allow easy tuning.
N_CORES = int(os.getenv("SPACY_N_PROCESS", "1"))

# === Logging setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def log_memory(prefix=""):
    mem = psutil.virtual_memory()
    logging.info(f"{prefix} Memory usage: {mem.percent:.1f}% used, {mem.available / (1024**3):.2f} GB available")

# === Timing decorator ===
def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        logging.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time()
        logging.info(f"Finished {func.__name__} in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# === Load SpaCy model ===
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_text(text: str) -> str:
    """Tokenize, lemmatize, remove stopwords/non-alpha tokens."""
    doc = nlp(text or "")
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

@track_time
def extract_text_features(data: pd.DataFrame, max_features: int = TEXT_FEATURES_LIMIT):
    """Vectorize preprocessed text from dataframe['description'] using multicore."""
    all_texts = data["description"].fillna("").tolist()
    processed_descriptions = []

    logging.info(f"Total samples: {len(all_texts)}, n_process={N_CORES}")
    log_memory("Before preprocessing")

    # Single nlp.pipe pass over all texts — avoids fork/join overhead of
    # per-batch loops while keeping memory bounded via spaCy's internal streaming.
    processed_descriptions = [
        " ".join(t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha)
        for doc in tqdm(
            nlp.pipe(all_texts, n_process=N_CORES, batch_size=BATCH_SIZE),
            total=len(all_texts), desc="SpaCy lemmatise",
        )
    ]
    log_memory("After SpaCy")

    data["processed_description"] = processed_descriptions
    logging.info("Vectorizing text features...")
    vectorizer = CountVectorizer(max_features=max_features)
    text_features = vectorizer.fit_transform(processed_descriptions).toarray().astype(np.float32)
    log_memory("After vectorization")
    return text_features, vectorizer


if __name__ == "__main__":
    logging.info("Loading training data...")
    df = pd.read_csv("data/X_train_update.csv")

    logging.info("Extracting text features...")
    features, vectorizer = extract_text_features(df)

    logging.info("Saving outputs...")
    os.makedirs("data/feature_cache", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    np.save(TEXT_FEATURES_FILE, features)
    with open(TEXT_VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    logging.info(f"Preprocessing complete: {TEXT_FEATURES_FILE}, {TEXT_VECTORIZER_FILE}")
