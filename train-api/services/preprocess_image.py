import os
import numpy as np
import pandas as pd
import logging
from time import time
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# === Constants ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32        # safe for standard RAM
SAVE_FILE = "data/feature_cache/image_features.npy"

# === Logging ===
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

# === Utility: track execution time ===
def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        logging.info(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# === Image feature extraction ===
@track_time
def extract_image_features(df, batch_size=BATCH_SIZE, save_file=SAVE_FILE):
    if df.empty:
        logging.warning("DataFrame is empty. Nothing to process.")
        return None

    logging.info(f"Starting image feature extraction: {len(df)} images")

    # Load pretrained model once
    try:
        base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                              input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    except Exception as e:
        logging.error(f"Failed to load ResNet50: {e}")
        return None

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    num_batches = (len(df) + batch_size - 1) // batch_size
    all_features = []

    for i in tqdm(range(num_batches), desc="Processing image batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        if batch_df.empty:
            continue

        try:
            generator = datagen.flow_from_dataframe(
                dataframe=batch_df,
                x_col="image_path",
                target_size=IMAGE_SIZE,
                batch_size=batch_size,
                class_mode=None,
                shuffle=False
            )
            batch_features = base_model.predict(generator, verbose=0).astype(np.float32)
            all_features.append(batch_features)
        except Exception as e:
            logging.error(f"Error processing batch {i} ({start_idx}-{end_idx}): {e}")
            continue

        logging.info(f"Processed batch {i+1}/{num_batches} ({start_idx}-{end_idx})")

    if all_features:
        image_features = np.concatenate(all_features, axis=0)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        np.save(save_file, image_features)
        logging.info(f"Saved image features: {save_file}")
        return save_file

    logging.warning("No features were extracted.")
    return None

if __name__ == "__main__":
    logging.info("Loading image metadata...")
    df_images = pd.read_csv("data/X_train_update.csv")
    
    # Construct full image paths
    image_folder = "data/images/image_train"
    df_images["image_path"] = df_images.apply(
        lambda row: os.path.join(image_folder, f"image_{row['imageid']}_product_{row['productid']}.jpg"), axis=1
    )

    missing_images = df_images[~df_images["image_path"].apply(os.path.exists)]
    logging.warning(f"Missing images: {len(missing_images)}")

    extract_image_features(df_images, batch_size=BATCH_SIZE)
