import os
import pandas as pd
from utils.logger import get_logger

# Make sure DATA_PATH is defined in your project (e.g., /app)
DATA_PATH = "/app"
logger = get_logger()

def load_and_merge_data(image_folder: str = None, use_dev_images: bool = None):
    """
    Load and merge X and Y CSVs, and add full image paths.

    Args:
        image_folder (str, optional): Folder inside /app/images containing images. 
                                      Options: 
                                      - 'images/image_sample' (development subset)
                                      - 'images/image_train' (full training set)
                                      - 'images/image_test' (validation/test set)
                                      If None, determined from use_dev_images or USE_DEV_IMAGES env variable.

        use_dev_images (bool, optional): Whether to use the development subset of images. 
                                         Overrides image_folder if provided.
                                         - True  -> 'images/image_sample'
                                         - False -> 'images/image_train'
                                         If None, fallback to USE_DEV_IMAGES env var.
        
    Returns:
        pd.DataFrame: Merged DataFrame with an additional 'image_path' column.
    """
    # Resolve which folder to use
    if image_folder is None:
        if use_dev_images is not None:
            image_folder = "images/image_sample" if use_dev_images else "images/image_train"
        else:
            use_dev = os.getenv("USE_DEV_IMAGES", "true").lower() == "true"
            image_folder = "images/image_sample" if use_dev else "images/image_train"

    IMAGE_PATH = os.path.join(DATA_PATH, image_folder)  # e.g., /app/images/image_train

    # Load CSVs
    X_train = pd.read_csv(os.path.join(DATA_PATH, "data/X_train_update.csv"))
    Y_train = pd.read_csv(os.path.join(DATA_PATH, "data/Y_train_CVw08PX.csv"))

    # Merge on index column
    train_data = pd.merge(X_train, Y_train, on="Unnamed: 0")
    train_data.rename(columns={"Unnamed: 0": "id"}, inplace=True)

    # Construct image filenames
    def construct_image_path(row):
        fname = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        full_path = os.path.join(IMAGE_PATH, fname)

        if not os.path.exists(full_path):
            if "image_sample" in image_folder:
                # Dev mode → warn and skip
                logger.warning(f"[DEV MODE] Missing image skipped: {full_path}")
                return None
            else:
                # Full training → strict error
                raise FileNotFoundError(f"[FULL TRAIN] Image not found: {full_path}")

        return full_path

    train_data["image_path"] = train_data.apply(construct_image_path, axis=1)

    # Drop rows with missing images (only happens in dev mode)
    if "image_sample" in image_folder:
        before = len(train_data)
        train_data = train_data.dropna(subset=["image_path"])
        after = len(train_data)
        logger.info(f"[DEV MODE] Dropped {before - after} rows with missing images")

    return train_data
