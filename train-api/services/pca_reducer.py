import os
import numpy as np
import pickle
from sklearn.decomposition import IncrementalPCA
import psutil
import logging
from time import time
from tqdm import tqdm

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def log_memory_usage(prefix=""):
    mem = psutil.virtual_memory()
    logging.info(f"{prefix} Memory usage: {mem.percent:.1f}% used, {mem.available / (1024**3):.2f} GB available")

# --- Timing decorator ---
def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        logging.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time()
        logging.info(f"Finished {func.__name__} in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@track_time
def reduce_features(
    text_features_path: str, image_features_path: str,
    n_components_img=384,
    n_components_text=None,   # explicit text PCA components; if None, derived from target_dim
    target_dim=5300,
    save_dir="data",
    initial_batch_size=1024,
    text_encoder="countvectorizer",
):
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Load features ---
    logging.info("Loading features from disk")
    text_features = np.load(text_features_path, mmap_mode='r').astype(np.float32)
    image_features = np.load(image_features_path, mmap_mode='r').astype(np.float32)
    logging.info(f"Text features shape: {text_features.shape}, Image features shape: {image_features.shape}")
    log_memory_usage("After loading features")

    # --- Incremental PCA for images ---
    logging.info("Starting Incremental PCA for images")
    # IncrementalPCA requires every partial_fit batch to have >= n_components samples.
    # If the dataset is small (e.g. dev mode), shrink n_components to fit.
    n_components_img = min(n_components_img, image_features.shape[0], image_features.shape[1])
    pca_img = IncrementalPCA(n_components=n_components_img)
    batch_size = initial_batch_size
    total_samples = image_features.shape[0]

    start_idx = 0
    while start_idx < total_samples:
        end_idx = min(start_idx + batch_size, total_samples)
        # Absorb a too-small trailing residue into the current batch so that
        # the last partial_fit always has >= n_components samples.
        remaining = total_samples - end_idx
        if 0 < remaining < n_components_img:
            end_idx = total_samples
        batch = image_features[start_idx:end_idx]
        try:
            pca_img.partial_fit(batch)
            if start_idx % (batch_size*5) == 0:
                log_memory_usage(f"After image batch {start_idx}-{end_idx}")
            start_idx = end_idx
        except MemoryError:
            batch_size = max(batch_size // 2, 16)
            logging.warning(f"MemoryError: reducing image batch size to {batch_size}")
            if batch_size < 16:
                raise MemoryError("Cannot fit even a single image batch into memory")

    logging.info("Transforming image features using PCA (batched)")
    image_features_reduced = np.vstack([
        pca_img.transform(image_features[i:i + batch_size])
        for i in range(0, image_features.shape[0], batch_size)
    ])
    logging.info(f"Image features reduced shape: {image_features_reduced.shape}")
    log_memory_usage("After image PCA transform")

    # --- Text features: PCA for CountVectorizer, pass-through for MiniLM / CLIP ---
    pca_text = None
    pca_text_path = None

    if text_encoder in ("minilm", "clip", "mpnet"):
        # MiniLM (384-d), CLIP (512-d), and mpnet (768-d) embeddings are already compact — no PCA needed
        logging.info(f"{text_encoder} encoder: skipping text PCA, using embeddings directly")
        text_features_reduced = text_features
        logging.info(f"{text_encoder} text features shape: {text_features_reduced.shape}")
    else:
        logging.info("Starting Incremental PCA for text")
        if n_components_text is not None:
            # Explicit value from params.yaml
            n_text_components = min(
                n_components_text,
                text_features.shape[0],
                text_features.shape[1],
            )
        else:
            # Legacy derivation from target_dim
            n_text_components = min(
                target_dim - image_features_reduced.shape[1],
                initial_batch_size,
                text_features.shape[0],
                text_features.shape[1],
            )
        # IncrementalPCA requires batch_size >= n_components — enlarge if needed
        batch_size = max(initial_batch_size, n_text_components)
        logging.info(f"Text PCA: n_components={n_text_components}, batch_size={batch_size}")
        pca_text = IncrementalPCA(n_components=n_text_components)
        total_samples = text_features.shape[0]

        start_idx = 0
        while start_idx < total_samples:
            end_idx = min(start_idx + batch_size, total_samples)
            remaining = total_samples - end_idx
            if 0 < remaining < n_text_components:
                end_idx = total_samples
            batch = text_features[start_idx:end_idx]
            try:
                pca_text.partial_fit(batch)
                if start_idx % (batch_size * 5) == 0:
                    log_memory_usage(f"After text batch {start_idx}-{end_idx}")
                start_idx = end_idx
            except MemoryError:
                batch_size = max(batch_size // 2, 16)
                logging.warning(f"MemoryError: reducing text batch size to {batch_size}")
                if batch_size < 16:
                    raise MemoryError("Cannot fit even a single text batch into memory")

        logging.info("Transforming text features using PCA (batched)")
        text_features_reduced = np.vstack([
            pca_text.transform(text_features[i:i + batch_size])
            for i in range(0, text_features.shape[0], batch_size)
        ])
        logging.info(f"Text features reduced shape: {text_features_reduced.shape}")
        log_memory_usage("After text PCA transform")

    # --- Combine features ---
    logging.info("Combining text and image features")
    X_reduced = np.hstack([text_features_reduced, image_features_reduced]).astype(np.float32)
    logging.info(f"Combined features shape: {X_reduced.shape}")
    log_memory_usage("After combining features")

    # --- Save outputs ---
    # Both pca_components AND encoder in filename so different PCA configs coexist on disk.
    # Switching pca_components no longer triggers a full 6-hour recompute — the right
    # versioned file is loaded directly on cache hit.
    # pca_image.pkl (unversioned) is also saved by save_artifacts for predict-api compatibility.
    X_reduced_path = os.path.join(save_dir, f"X_reduced_{text_encoder}_{n_components_img}.npy")
    pca_img_path   = os.path.join(save_dir, f"pca_image_{n_components_img}.pkl")

    np.save(X_reduced_path, X_reduced)
    with open(pca_img_path, "wb") as f:
        pickle.dump(pca_img, f)

    if pca_text is not None:
        pca_text_path = os.path.join(save_dir, f"pca_text_{n_text_components}.pkl")
        with open(pca_text_path, "wb") as f:
            pickle.dump(pca_text, f)
        logging.info(f"Saved PCA text model -> {pca_text_path}")

    logging.info(f"Saved X_reduced -> {X_reduced_path}")
    logging.info(f"Saved PCA image model -> {pca_img_path}")

    # Explicitly free large arrays before returning so glibc releases them
    # via munmap() (mmap threshold) rather than tcache, preventing double-free
    # when TensorFlow's allocator initialises in the same process shortly after.
    import gc
    del text_features, image_features, X_reduced
    del image_features_reduced, text_features_reduced
    gc.collect()
    logging.info("PCA arrays freed, gc collected")

    return X_reduced_path, pca_img_path, pca_text_path

if __name__ == "__main__":
    # FIX 3: Update save_dir to "artifacts" to match DVC expectations
    reduce_features(
        text_features_path="data/feature_cache/text_features.npy",
        image_features_path="data/feature_cache/image_features.npy",
        n_components_img=384,
        target_dim=5300,
        save_dir="artifacts",  # Changed from "data" to "artifacts"
        initial_batch_size=1024
    )