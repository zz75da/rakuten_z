"""
One-shot OCR extraction for Rakuten product images.

Run once to build the cache; subsequent pipeline runs load from CSV.
Results are saved incrementally so the job can be interrupted and resumed.

Usage (inside train-api container, or standalone):
    python services/run_ocr_extraction.py [--dev] [--resume]

    --dev     : use images/image_sample (dev subset, ~1k images, useful for testing)
    --resume  : skip images already present in the output CSV (default behaviour)

Output:
    /app/data/feature_cache/ocr_text.csv
    Columns: imageid (int), productid (int), ocr_text (str, may be empty)

After completion the script deletes the stale text-feature cache so the next
pipeline run regenerates text_features.npy with OCR text included.
"""
import os
import sys
import csv
import json
import time
import logging
import argparse
import traceback

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = "/app"
OCR_CACHE   = "/app/data/feature_cache/ocr_text.csv"
PROGRESS_TMP = "/app/data/feature_cache/ocr_progress.json"

# Stale caches that must be removed so the next train regenerates with OCR text
_STALE_ON_COMPLETE = [
    "/app/data/feature_cache/text_features.npy",
    "/app/data/artifacts/text_vectorizer.pkl",
    "/app/data/feature_cache/text_features_meta.json",
]


# ---------------------------------------------------------------------------
# Image preprocessing — improves Tesseract accuracy on product photos
# ---------------------------------------------------------------------------
def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Grayscale + upscale small images + mild sharpening.
    Tesseract performs best on ~300 DPI images; product thumbnails are often
    small (< 224×224) so we upscale to at least 300 px on the short side.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    min_side = min(h, w)
    if min_side < 300:
        scale = 300 / min_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # Light sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    return gray


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
def _ocr_image(image_path: str) -> str:
    """Return cleaned OCR text for one image, or '' on any error."""
    try:
        import pytesseract
        img = cv2.imread(image_path)
        if img is None:
            return ""
        processed = _preprocess_for_ocr(img)
        # psm 11: sparse text (product images have scattered text, no clear layout)
        # oem 1:  LSTM engine only (faster + more accurate than legacy)
        raw = pytesseract.image_to_string(
            processed,
            lang="fra+eng",
            config="--psm 11 --oem 1",
        )
        # Strip whitespace / control chars; collapse runs of spaces
        cleaned = " ".join(raw.split())
        return cleaned
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_ocr(use_dev: bool = False, resume: bool = True):
    os.makedirs("/app/data/feature_cache", exist_ok=True)

    # ---- load image manifest ----
    image_folder = "images/image_sample" if use_dev else "images/image_train"
    image_dir    = os.path.join(DATA_PATH, image_folder)

    X_csv = os.getenv("TRAIN_CSV_X_PATH", "/app/data/X_train_update.csv")
    X = pd.read_csv(X_csv)
    X.rename(columns={"Unnamed: 0": "id"}, inplace=True)

    try:
        existing_files = {e.name for e in os.scandir(image_dir) if e.is_file()}
    except FileNotFoundError:
        log.error(f"Image directory not found: {image_dir}")
        sys.exit(1)

    X["_fname"] = X.apply(
        lambda r: f"image_{r['imageid']}_product_{r['productid']}.jpg", axis=1
    )
    X = X[X["_fname"].isin(existing_files)].reset_index(drop=True)
    log.info(f"Images on disk: {len(X):,}  (mode={'DEV' if use_dev else 'FULL'})")

    # ---- resume: load already-processed image ids ----
    already_done: set = set()
    if resume and os.path.exists(OCR_CACHE):
        try:
            done_df = pd.read_csv(OCR_CACHE, dtype={"imageid": int, "productid": int})
            already_done = set(zip(done_df["imageid"], done_df["productid"]))
            log.info(f"Resuming — {len(already_done):,} images already cached")
        except Exception as e:
            log.warning(f"Could not read existing cache for resume: {e}")

    rows_to_process = X[
        ~X.apply(lambda r: (int(r["imageid"]), int(r["productid"])) in already_done,
                 axis=1)
    ].reset_index(drop=True)

    total       = len(rows_to_process)
    total_all   = len(X)
    log.info(f"Images to process: {total:,}  (skipping {len(already_done):,} cached)")

    if total == 0:
        log.info("Nothing to do — OCR cache is already complete.")
        return

    # ---- open CSV in append mode (resume-safe) ----
    write_header = not os.path.exists(OCR_CACHE) or len(already_done) == 0
    csv_file = open(OCR_CACHE, "a", newline="", encoding="utf-8")
    writer   = csv.writer(csv_file)
    if write_header:
        csv_file.seek(0)
        csv_file.truncate()
        writer.writerow(["imageid", "productid", "ocr_text"])

    # ---- process ----
    t0        = time.time()
    FLUSH_N   = 50   # flush to disk every N images (checkpoint)
    done      = 0
    errors    = 0

    for idx, row in rows_to_process.iterrows():
        img_path = os.path.join(image_dir, row["_fname"])
        ocr_text = _ocr_image(img_path)
        if not ocr_text:
            errors += 1

        writer.writerow([int(row["imageid"]), int(row["productid"]), ocr_text])
        done += 1

        if done % FLUSH_N == 0:
            csv_file.flush()

        if done % 500 == 0 or done == total:
            elapsed  = time.time() - t0
            rate     = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            pct      = (len(already_done) + done) / total_all * 100
            log.info(
                f"  [{pct:.1f}%] {len(already_done)+done:,}/{total_all:,}  "
                f"rate={rate:.1f} img/s  ETA={remaining/3600:.1f}h  "
                f"empty={errors}"
            )
            # write progress sidecar
            with open(PROGRESS_TMP, "w") as pf:
                json.dump({
                    "done": len(already_done) + done,
                    "total": total_all,
                    "errors": errors,
                    "eta_hours": round(remaining / 3600, 2),
                }, pf)

    csv_file.flush()
    csv_file.close()

    elapsed = time.time() - t0
    log.info(
        f"OCR complete — {total_all:,} images, {errors} empty results, "
        f"elapsed {elapsed/3600:.1f}h → {OCR_CACHE}"
    )

    # ---- invalidate stale text caches so next train regenerates ----
    for stale in _STALE_ON_COMPLETE:
        if os.path.exists(stale):
            os.remove(stale)
            log.info(f"Deleted stale cache: {stale}")

    if os.path.exists(PROGRESS_TMP):
        os.remove(PROGRESS_TMP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev",    action="store_true", help="Use image_sample (dev subset)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Reprocess all images, ignoring existing cache")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    run_ocr(use_dev=args.dev, resume=args.resume)
