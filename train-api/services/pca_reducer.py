import os
import numpy as np
import pickle
from sklearn.decomposition import IncrementalPCA
import psutil
import logging
from time import time
from tqdm import tqdm

os.makedirs("artifacts", exist_ok=True)
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
    n_components_img=300,
    target_dim=5300,
    save_dir="data",
    initial_batch_size=1024
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
    pca_img = IncrementalPCA(n_components=n_components_img)  # Variable named pca_img
    batch_size = initial_batch_size
    total_samples = image_features.shape[0]

    start_idx = 0
    while start_idx < total_samples:
        end_idx = min(start_idx + batch_size, total_samples)
        batch = image_features[start_idx:end_idx]
        try:
            pca_img.partial_fit(batch)
            if start_idx % (batch_size*5) == 0:
                log_memory_usage(f"After image batch {start_idx}-{end_idx}")
            start_idx += batch_size
        except MemoryError:
            batch_size = max(batch_size // 2, 16)
            logging.warning(f"MemoryError: reducing image batch size to {batch_size}")
            if batch_size < 16:
                raise MemoryError("Cannot fit even a single image batch into memory")

    logging.info("Transforming image features using PCA")
    image_features_reduced = pca_img.transform(image_features)
    logging.info(f"Image features reduced shape: {image_features_reduced.shape}")
    log_memory_usage("After image PCA transform")

    # --- Incremental PCA for text ---
    logging.info("Starting Incremental PCA for text")
    n_text_components = min(target_dim - image_features_reduced.shape[1], initial_batch_size)
    pca_text = IncrementalPCA(n_components=n_text_components)
    batch_size = initial_batch_size
    total_samples = text_features.shape[0]

    start_idx = 0
    while start_idx < total_samples:
        end_idx = min(start_idx + batch_size, total_samples)
        batch = text_features[start_idx:end_idx]
        try:
            pca_text.partial_fit(batch)
            if start_idx % (batch_size*5) == 0:
                log_memory_usage(f"After text batch {start_idx}-{end_idx}")
            start_idx += batch_size
        except MemoryError:
            batch_size = max(batch_size // 2, 16)
            logging.warning(f"MemoryError: reducing text batch size to {batch_size}")
            if batch_size < 16:
                raise MemoryError("Cannot fit even a single text batch into memory")

    logging.info("Transforming text features using PCA")
    text_features_reduced = pca_text.transform(text_features)
    logging.info(f"Text features reduced shape: {text_features_reduced.shape}")
    log_memory_usage("After text PCA transform")

    # --- Combine features ---
    logging.info("Combining text and image features")
    X_reduced = np.hstack([text_features_reduced, image_features_reduced]).astype(np.float32)
    logging.info(f"Combined features shape: {X_reduced.shape}")
    log_memory_usage("After combining features")

    # --- Save outputs ---
    # FIX 1: Use consistent save_dir parameter for all outputs
    X_reduced_path = os.path.join(save_dir, "X_reduced.npy")  # Changed to use save_dir parameter
    pca_img_path = os.path.join(save_dir, "pca_image.pkl")
    pca_text_path = os.path.join(save_dir, "pca_text.pkl")

    np.save(X_reduced_path, X_reduced)
    with open(pca_img_path, "wb") as f:
        pickle.dump(pca_img, f)  # Variable is pca_img, not pca_image
    with open(pca_text_path, "wb") as f:
        pickle.dump(pca_text, f)

    # FIX 2: REMOVE THESE DUPLICATE LINES - they're causing the NameError
    # pickle.dump(pca_image, open("artifacts/pca_image.pkl", "wb"))  # ❌ DUPLICATE & WRONG VARIABLE NAME
    # pickle.dump(pca_text, open("artifacts/pca_text.pkl", "wb"))   # ❌ DUPLICATE

    logging.info(f"Saved X_reduced -> {X_reduced_path}")
    logging.info(f"Saved PCA image model -> {pca_img_path}")
    logging.info(f"Saved PCA text model -> {pca_text_path}")

    return X_reduced_path, pca_img_path, pca_text_path

if __name__ == "__main__":
    # FIX 3: Update save_dir to "artifacts" to match DVC expectations
    reduce_features(
        text_features_path="data/text_features.npy",
        image_features_path="data/image_features.npy",
        n_components_img=300,
        target_dim=5300,
        save_dir="artifacts",  # Changed from "data" to "artifacts"
        initial_batch_size=1024
    )