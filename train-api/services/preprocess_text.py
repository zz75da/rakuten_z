import html as _html
import logging
import os
import re as _re

import numpy as np
import pandas as pd
import psutil
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

# === Constants ===
TEXT_FEATURES_FILE   = "data/feature_cache/text_features.npy"
TEXT_VECTORIZER_FILE = "data/text_vectorizer.pkl"
OCR_CACHE            = "/app/data/feature_cache/ocr_text.csv"
BATCH_SIZE = 5000
N_CORES    = int(os.getenv("SPACY_N_PROCESS", "1"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def log_memory(prefix=""):
    mem = psutil.virtual_memory()
    logging.info(f"{prefix} Memory: {mem.percent:.1f}% used, {mem.available / 1024**3:.2f} GB free")

def track_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        logging.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Finished {func.__name__} in {time() - start:.2f}s")
        return result
    return wrapper

# === Load French spaCy model (primary: handles 49% French Rakuten data correctly) ===
try:
    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
    logging.info("Loaded fr_core_news_sm (French lemmatization + French stopwords)")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        logging.warning("fr_core_news_sm not found, falling back to en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# === Multilingual stopwords (French + English) added to spaCy's built-in list ===
# French stopwords missing from en_core_web_sm that pollute the vocabulary
_EXTRA_STOPS = {
    # French
    "le","la","les","de","du","des","un","une","en","et","au","aux",
    "pour","avec","sur","par","ce","se","sa","son","ses","qui","que",
    "dans","cette","est","sont","plus","mais","ou","ni","car","donc",
    "or","ne","pas","si","tout","bien","très","même","autre","aussi",
    "leur","leurs","nous","vous","ils","elles","je","tu","il","elle",
    "mon","ton","ma","ta","mes","tes","nos","vos","eux",
    "être","avoir","faire","aller","dire","voir","vouloir","pouvoir",
    # English (supplement spaCy's list for English product titles)
    "the","a","an","of","in","is","are","was","were","and","or","but",
    "for","with","on","at","to","from","by","as","into","about","up",
    "out","after","before","between","through","during","its","it",
}
nlp.Defaults.stop_words.update(_EXTRA_STOPS)
for w in _EXTRA_STOPS:
    nlp.vocab[w].is_stop = True


def _clean_text(text) -> str:
    """HTML-unescape, strip tags, normalise whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _html.unescape(text)
    text = _re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def _build_combined_text(data: pd.DataFrame) -> list[str]:
    """
    Concatenate designation + description + OCR text (if cache exists).

    Sources:
      - designation : 100% populated product title
      - description : available for ~65% of products (35% null, HTML entities)
      - ocr_text    : text extracted from product images by run_ocr_extraction.py
                      (optional — pipeline works without it; run once to populate)
    """
    desig = data["designation"].fillna("").apply(_clean_text)
    descr = data["description"].fillna("").apply(_clean_text)
    combined = desig + " " + descr

    if os.path.exists(OCR_CACHE):
        try:
            ocr_df = pd.read_csv(
                OCR_CACHE,
                dtype={"imageid": int, "productid": int, "ocr_text": str},
            ).fillna("")
            merged = data[["imageid", "productid"]].reset_index(drop=True).merge(
                ocr_df[["imageid", "productid", "ocr_text"]],
                on=["imageid", "productid"],
                how="left",
            )
            ocr_col = merged["ocr_text"].fillna("").apply(_clean_text)
            combined = combined + " " + ocr_col
            n_nonempty = (ocr_col.str.strip() != "").sum()
            logging.info(
                f"OCR text merged from {OCR_CACHE} "
                f"({n_nonempty:,}/{len(data):,} images had extractable text)"
            )
        except Exception as e:
            logging.warning(f"OCR cache read failed, skipping OCR text: {e}")

    return combined.str.strip().tolist()


@track_time
def extract_text_features(data: pd.DataFrame, max_features: int = 10000, fit_only: bool = False):
    """
    Vectorize product text using French spaCy lemmatization + TF-IDF (TfidfVectorizer).

    fit_only=True: fits and returns the vectorizer but does NOT create the dense feature
    matrix. Use when text_features.npy already exists on disk but text_vectorizer.pkl
    was corrupted or deleted — avoids the 3.4 GB OOM that full regeneration would cause.
    Returns (None, vectorizer) when fit_only=True.
    """
    all_texts = _build_combined_text(data)
    n = len(all_texts)
    logging.info(f"Total samples: {n}, n_process={N_CORES}, fit_only={fit_only}")
    log_memory("Before preprocessing")

    def _lemmatize(doc):
        return " ".join(
            t.lemma_.lower()
            for t in doc
            if not t.is_stop       # removes French + English stopwords
            and not t.is_punct
            and not t.is_space
            and (t.is_alpha or t.like_num)  # keep words AND numbers (model codes, sizes)
            and len(t.text) > 1            # skip single-character tokens
        )

    processed = [
        _lemmatize(doc)
        for doc in nlp.pipe(all_texts, n_process=N_CORES, batch_size=BATCH_SIZE)
    ]

    log_memory("After SpaCy")
    data = data.copy()
    data["processed_description"] = processed

    logging.info(f"Vectorizing with TfidfVectorizer(max_features={max_features}, ngram_range=(1,2), sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # unigrams + bigrams: "jeu", "jeu video", "livre enfant"
        min_df=2,             # ignore tokens appearing in only 1 document
        sublinear_tf=True,    # apply log(1+tf) instead of raw tf — compresses high-freq tokens
    )

    if fit_only:
        # fit() only — vocabulary built but no dense matrix created.
        # Peak memory: sparse CSR matrix (~100 MB) vs toarray() → 3.4 GB.
        vectorizer.fit(processed)
        log_memory("After fit-only vectorization")
        return None, vectorizer

    text_features = vectorizer.fit_transform(processed).toarray().astype(np.float32)
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
    import pickle
    with open(TEXT_VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    logging.info(f"Done: {features.shape} → {TEXT_FEATURES_FILE}")
