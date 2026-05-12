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
MINILM_TEXT_FEATURES_FILE = "data/feature_cache/text_features_minilm.npy"
TEXT_VECTORIZER_FILE = "data/text_vectorizer.pkl"
MINILM_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 5000  # adjust based on memory
N_CORES = 6  # your Core i7

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

    logging.info(f"Total samples: {len(all_texts)}")
    log_memory("Before preprocessing")

    # Process in batches to avoid memory overload
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Text Preprocessing Batches"):
        batch = all_texts[i:i+BATCH_SIZE]
        processed_batch = [
            doc for doc in nlp.pipe(batch, n_process=N_CORES, batch_size=1000)
        ]
        processed_batch = [
            " ".join([t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha])
            for doc in processed_batch
        ]
        processed_descriptions.extend(processed_batch)
        log_memory(f"After processing batch {i}-{i+len(batch)}")

    data["processed_description"] = processed_descriptions
    logging.info("Vectorizing text features...")
    vectorizer = CountVectorizer(max_features=max_features)
    text_features = vectorizer.fit_transform(processed_descriptions).toarray().astype(np.float32)
    log_memory("After vectorization")
    return text_features, vectorizer


@track_time
def extract_text_features_minilm(data: pd.DataFrame):
    """Encode text using SentenceTransformer MiniLM — no preprocessing needed."""
    from sentence_transformers import SentenceTransformer
    logging.info(f"Loading SentenceTransformer: {MINILM_MODEL_NAME}")
    encoder = SentenceTransformer(MINILM_MODEL_NAME)
    texts = data["description"].fillna("").tolist()
    logging.info(f"Encoding {len(texts)} texts with MiniLM...")
    log_memory("Before MiniLM encoding")
    embeddings = encoder.encode(
        texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True
    )
    log_memory("After MiniLM encoding")
    logging.info(f"MiniLM embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32), encoder


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
