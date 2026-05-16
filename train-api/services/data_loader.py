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

    # Construct image filenames.
    # Missing files are tolerated in both DEV and FULL modes: the row is
    # dropped and a warning is logged. This avoids a hard failure when the
    # user has manually pruned images from image_train/ (or image_sample/).
    mode_label = "DEV" if "image_sample" in image_folder else "FULL"

    # Build existence set with a single os.scandir call (one syscall) instead of
    # one os.path.exists() per row (~85 k individual stat syscalls).
    try:
        existing_files = {e.name for e in os.scandir(IMAGE_PATH) if e.is_file()}
    except FileNotFoundError:
        existing_files = set()

    def construct_image_path(row):
        fname = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        if fname not in existing_files:
            return None
        return os.path.join(IMAGE_PATH, fname)

    train_data["image_path"] = train_data.apply(construct_image_path, axis=1)

    # Log a sample of missing image names (not all 85k if entire folder is wrong)
    missing_mask = train_data["image_path"].isna()
    if missing_mask.any():
        sample = train_data.loc[missing_mask, ["imageid", "productid"]].head(3)
        for _, r in sample.iterrows():
            logger.warning(f"[{mode_label} MODE] Missing image: image_{r['imageid']}_product_{r['productid']}.jpg")

    # Drop rows whose image is missing on disk.
    before = len(train_data)
    train_data = train_data.dropna(subset=["image_path"]).reset_index(drop=True)
    after = len(train_data)
    dropped = before - after
    if dropped:
        ratio = dropped / before
        logger.info(
            f"[{mode_label} MODE] Dropped {dropped}/{before} rows with missing images "
            f"({ratio:.1%})"
        )
        # Hard fail only if effectively NO image is on disk (catches misconfigured mount).
        if after == 0:
            raise FileNotFoundError(
                f"[{mode_label} MODE] No image found in {IMAGE_PATH}. "
                f"Check the volume mount and that the folder contains the expected images."
            )

    return train_data
