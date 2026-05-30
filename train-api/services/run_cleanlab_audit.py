"""
Cleanlab label-quality audit for Rakuten training data.

Runs once after training to flag likely mislabeled samples using confident learning.
Predictions are made on the full training set using the best available model.

Usage (inside train-api container):
    python services/run_cleanlab_audit.py [--encoder clip|cv|minilm|mpnet]

Output:
    /app/data/artifacts/cleanlab_report_<encoder>.csv
    Columns: idx, imageid, productid, designation, true_label, true_class,
             predicted_label, predicted_class, confidence, issue_type

Memory budget: ~500 MB peak (X_reduced ~290 MB + model ~50 MB + pred_probs ~9 MB)
Disk budget  : ~2 MB per report CSV
"""
import os
import sys
import gc
import argparse
import pickle
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, "/app")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

ARTIFACTS  = "/app/data/artifacts"
FEAT_CACHE = "/app/data/feature_cache"
DATA_PATH  = "/app/data"

_ENCODER_FILES = {
    "clip":             (f"{ARTIFACTS}/X_reduced_clip.npy",          f"{ARTIFACTS}/neural_network_model_clip.keras"),
    "minilm":           (f"{ARTIFACTS}/X_reduced_minilm.npy",        f"{ARTIFACTS}/neural_network_model_minilm.keras"),
    "mpnet":            (f"{ARTIFACTS}/X_reduced_mpnet.npy",         f"{ARTIFACTS}/neural_network_model_mpnet.keras"),
    "countvectorizer":  (f"{ARTIFACTS}/X_reduced_countvectorizer.npy", f"{ARTIFACTS}/neural_network_model.keras"),
}


def run_audit(encoder: str = "clip", batch_size: int = 512):
    log.info(f"Cleanlab audit — encoder={encoder}")

    # ── Validate inputs ──────────────────────────────────────────────────────
    if encoder not in _ENCODER_FILES:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose: {list(_ENCODER_FILES)}")

    x_path, model_path = _ENCODER_FILES[encoder]
    label_enc_path = os.path.join(ARTIFACTS, "label_encoder.pkl")
    x_csv = os.getenv("TRAIN_CSV_X_PATH", f"{DATA_PATH}/X_train_update.csv")
    y_csv = os.getenv("TRAIN_CSV_Y_PATH", f"{DATA_PATH}/Y_train_CVw08PX.csv")

    for p, name in [(x_path, "X_reduced"), (model_path, "model"),
                    (label_enc_path, "label_encoder"), (x_csv, "X_csv"), (y_csv, "Y_csv")]:
        if not os.path.exists(p):
            log.error(f"Required file not found: {p} ({name})")
            sys.exit(1)

    out_path = os.path.join(ARTIFACTS, f"cleanlab_report_{encoder}.csv")

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading labels and metadata...")
    X_df = pd.read_csv(x_csv)
    Y_df = pd.read_csv(y_csv)
    X_df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
    Y_df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
    data = X_df.merge(Y_df, on="id")

    with open(label_enc_path, "rb") as f:
        label_enc = pickle.load(f)

    labels = label_enc.transform(data["prdtypecode"].values)
    n_samples = len(labels)
    log.info(f"Samples: {n_samples:,}  |  Classes: {len(label_enc.classes_)}")

    # ── Load features in one shot (pre-validated size) ────────────────────────
    file_size_mb = os.path.getsize(x_path) / 1024**2
    log.info(f"Loading X_reduced ({file_size_mb:.0f} MB)...")
    X = np.load(x_path)

    if len(X) != n_samples:
        log.error(f"Shape mismatch: X has {len(X)} rows but labels have {n_samples}")
        sys.exit(1)

    # ── Load model and predict in batches (memory-safe) ──────────────────────
    log.info("Loading model...")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    model = tf.keras.models.load_model(model_path, compile=False)

    log.info(f"Predicting on {n_samples:,} samples (batch_size={batch_size})...")
    pred_probs = np.zeros((n_samples, len(label_enc.classes_)), dtype=np.float32)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        pred_probs[start:end] = model.predict(X[start:end], verbose=0)
        if start % 10000 == 0 and start > 0:
            log.info(f"  {start:,}/{n_samples:,}")

    # Free large arrays immediately
    del X
    del model
    gc.collect()

    # ── Run Cleanlab ──────────────────────────────────────────────────────────
    log.info("Running cleanlab find_label_issues...")
    from cleanlab.filter import find_label_issues
    issue_mask = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
        filter_by="prune_by_noise_rate",
    )

    n_issues = len(issue_mask)
    pct = n_issues / n_samples * 100
    log.info(f"Label issues found: {n_issues:,} / {n_samples:,} ({pct:.1f}%)")

    # ── Build report ──────────────────────────────────────────────────────────
    log.info("Building report...")
    pred_labels  = np.argmax(pred_probs, axis=1)
    confidence   = pred_probs[np.arange(n_samples), pred_labels]

    issue_df = data.iloc[issue_mask].copy().reset_index(drop=True)
    issue_df["true_label"]       = labels[issue_mask]
    issue_df["true_class"]       = label_enc.inverse_transform(labels[issue_mask])
    issue_df["predicted_label"]  = pred_labels[issue_mask]
    issue_df["predicted_class"]  = label_enc.inverse_transform(pred_labels[issue_mask])
    issue_df["confidence"]       = np.round(confidence[issue_mask], 4)

    keep_cols = ["id", "imageid", "productid", "designation",
                 "true_label", "true_class", "predicted_label", "predicted_class", "confidence"]
    keep_cols = [c for c in keep_cols if c in issue_df.columns]
    issue_df  = issue_df[keep_cols].sort_values("confidence")

    issue_df.to_csv(out_path, index=False)
    log.info(f"Report saved -> {out_path}  ({os.path.getsize(out_path)//1024} KB)")

    # ── Per-class summary ─────────────────────────────────────────────────────
    log.info("\nTop 10 classes with most label issues:")
    class_counts = issue_df["true_class"].value_counts().head(10)
    for cls, cnt in class_counts.items():
        total_in_class = (data["prdtypecode"] == cls).sum()
        log.info(f"  class {cls}: {cnt} issues / {total_in_class} samples ({cnt/total_in_class*100:.1f}%)")

    del pred_probs
    gc.collect()
    return out_path, n_issues, n_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="clip",
                        choices=list(_ENCODER_FILES),
                        help="Which trained model to use for predictions (default: clip = best accuracy)")
    args = parser.parse_args()
    run_audit(encoder=args.encoder)
