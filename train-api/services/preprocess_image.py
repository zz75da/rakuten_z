import math
import os
import numpy as np
import pandas as pd
import logging
from time import time
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# === Constants ===
IMAGE_SIZE  = (224, 224)
# 64 is a safe default on a 16GB laptop: per-batch image decode ≈ 9 MB,
# TF activation buffers are modest, and np.concatenate double-allocation is
# eliminated (vs the old per-batch predict approach which peaked at ~1.4 GB).
# Tune up via IMAGE_BATCH_SIZE env var if RAM allows (128 → ~18 MB per batch).
BATCH_SIZE  = int(os.getenv("IMAGE_BATCH_SIZE", "64"))
SAVE_FILE   = "data/feature_cache/image_features.npy"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def track_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} finished in {time() - start:.2f}s")
        return result
    return wrapper


@track_time
def extract_image_features(df, batch_size=BATCH_SIZE, save_file=SAVE_FILE):
    if df.empty:
        logging.warning("DataFrame is empty — nothing to process.")
        return None

    n = len(df)
    logging.info(f"Starting image feature extraction: {n} images (batch_size={batch_size})")

    try:
        base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                              input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    except Exception as e:
        logging.error(f"Failed to load ResNet50: {e}")
        return None

    # Single generator over the full dataframe — one predict() call instead of
    # one per batch. Eliminates ~2 600 generator constructions and predict
    # overhead for a 84 k-image dataset.
    datagen   = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="image_path",
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )
    steps = math.ceil(n / batch_size)
    logging.info(f"Running ResNet50 inference: {steps} steps × {batch_size} images")

    try:
        image_features = base_model.predict(generator, steps=steps, verbose=1).astype(np.float32)
    except Exception as e:
        logging.error(f"ResNet50 predict failed: {e}")
        return None

    # Trim to exact length (generator may pad the last batch)
    image_features = image_features[:n]
    logging.info(f"Feature matrix shape: {image_features.shape}")

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    np.save(save_file, image_features)
    logging.info(f"Saved image features: {save_file}")
    return save_file


if __name__ == "__main__":
    logging.info("Loading image metadata...")
    df_images = pd.read_csv("data/X_train_update.csv")
    image_folder = "data/images/image_train"
    df_images["image_path"] = df_images.apply(
        lambda row: os.path.join(image_folder, f"image_{row['imageid']}_product_{row['productid']}.jpg"),
        axis=1,
    )
    missing = df_images[~df_images["image_path"].apply(os.path.exists)]
    if len(missing):
        logging.warning(f"Missing images: {len(missing)}")
    extract_image_features(df_images, batch_size=BATCH_SIZE)
