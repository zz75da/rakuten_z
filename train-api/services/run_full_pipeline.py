"""
Full training pipeline subprocess — invoked by app.py's _run_training_pipeline thread.

All TF imports happen in this process so the main uvicorn process heap is never
exposed to TF's allocator hooks.  Writes job status to /app/data/jobs/<job_id>.json.

Usage (called by app.py via subprocess.run):
    python services/run_full_pipeline.py <job_id> <text_encoder> <epochs> <batch_size> <use_dev_images> <use_cache>
"""
import sys
import os
import json
import pickle
import traceback
from datetime import datetime, timezone

sys.path.insert(0, "/app")

JOB_DIR = "/app/data/jobs"
os.makedirs(JOB_DIR, exist_ok=True)

job_id         = sys.argv[1]
text_encoder   = sys.argv[2]
epochs         = int(sys.argv[3])
batch_size     = int(sys.argv[4])
use_dev_images = sys.argv[5].lower() == "true"
use_cache      = sys.argv[6].lower() == "true"


def _write(status: dict):
    with open(os.path.join(JOB_DIR, f"{job_id}.json"), "w") as f:
        json.dump(status, f, default=str)


_STARTED_AT = datetime.now(timezone.utc).isoformat()
_write({"job_id": job_id, "status": "running", "step": "starting", "started_at": _STARTED_AT})


def _write_step(step: str, extra: dict = None):
    payload = {"job_id": job_id, "status": "running", "step": step, "started_at": _STARTED_AT}
    if extra:
        payload.update(extra)
    _write(payload)


try:
    import numpy as np
    import requests
    import subprocess as _sub

    from services.data_loader import load_and_merge_data
    from services.preprocess_text import extract_text_features
    from services.trainer import build_and_train_model
    from services.artifacts import save_artifacts

    X_CSV = os.getenv("TRAIN_CSV_X_PATH", "/app/data/X_train_update.csv")
    Y_CSV = os.getenv("TRAIN_CSV_Y_PATH", "/app/data/Y_train_CVw08PX.csv")

    # Read pipeline params from params.yaml (falls back to defaults if file absent)
    def _read_pipeline_params() -> dict:
        for path in ["/app/params.yaml", "params.yaml"]:
            if os.path.exists(path):
                try:
                    import yaml
                    with open(path) as _f:
                        p = yaml.safe_load(_f) or {}
                    pre = p.get("preprocess", {})
                    return {
                        "pca_components":       int(pre.get("pca_components", 300)),
                        "n_text_pca_components": int(pre.get("n_text_pca_components", 1024)),
                        "cv_max_features":       int(pre.get("cv_max_features", 5000)),
                    }
                except Exception:
                    pass
        return {"pca_components": 300, "n_text_pca_components": 1024, "cv_max_features": 5000}

    _pipeline_params     = _read_pipeline_params()
    pca_components       = _pipeline_params["pca_components"]
    n_text_pca_components = _pipeline_params["n_text_pca_components"]
    cv_max_features      = _pipeline_params["cv_max_features"]

    # --- Load data ---
    _write_step("loading_data")
    train_data = load_and_merge_data(use_dev_images=use_dev_images)
    os.makedirs("/app/data/feature_cache", exist_ok=True)

    IMAGE_CACHE = "/app/data/feature_cache/image_features.npy"

    # --- Text features ---
    _write_step("text_features", {"dataset_size": len(train_data)})
    if text_encoder == "minilm":
        TEXT_CACHE = "/app/data/feature_cache/text_features_minilm.npy"
        if not os.path.exists(TEXT_CACHE):
            raise RuntimeError(
                "MiniLM feature cache not found. "
                "Run: docker-compose --profile minilm run --rm minilm-encoder"
            )
        text_vectorizer = None
    else:
        TEXT_CACHE = "/app/data/feature_cache/text_features.npy"
        VECTORIZER_CACHE = "/app/data/artifacts/text_vectorizer.pkl"

        def _text_cache_valid():
            if not (os.path.exists(TEXT_CACHE) and os.path.exists(VECTORIZER_CACHE)):
                return False
            try:
                with open(VECTORIZER_CACHE, "rb") as _f:
                    _vec = pickle.load(_f)
                cached_vocab = len(_vec.vocabulary_)
                if cached_vocab != cv_max_features:
                    print(f"Text cache stale: vocab={cached_vocab} != requested {cv_max_features}")
                    return False
                return True
            except Exception:
                return False

        if use_cache and _text_cache_valid():
            with open(VECTORIZER_CACHE, "rb") as f:
                text_vectorizer = pickle.load(f)
        else:
            text_features, text_vectorizer = extract_text_features(
                train_data, max_features=cv_max_features
            )
            np.save(TEXT_CACHE, text_features)

    # --- Image features ---
    _write_step("image_features")
    if use_cache and os.path.exists(IMAGE_CACHE):
        image_features_path = IMAGE_CACHE
    else:
        from services.preprocess_image import extract_image_features
        image_features_path = extract_image_features(train_data)
        if not image_features_path:
            raise RuntimeError("Image feature extraction produced no features")

    # --- PCA in its own subprocess (prevents allocator overlap with numpy) ---
    # Output written to a temp file — not stdout — so any logging from pca_reducer
    # cannot corrupt the JSON result.
    #
    # Cache check: if X_reduced.npy and the PCA pkl files are already on disk AND
    # both feature inputs are newer than X_reduced (mtime), skip the subprocess.
    # This saves 20-40 min on every re-run where features haven't changed.
    _ARTIFACTS = "/app/data/artifacts"
    _X_reduced_cached  = os.path.join(_ARTIFACTS, "X_reduced.npy")
    _pca_img_cached    = os.path.join(_ARTIFACTS, "pca_image.pkl")
    _pca_text_cached   = os.path.join(_ARTIFACTS, "pca_text.pkl")   if text_encoder != "minilm" else None

    def _pca_cache_valid():
        required = [_X_reduced_cached, _pca_img_cached]
        if text_encoder != "minilm":
            required.append(_pca_text_cached)
        if not all(os.path.exists(p) for p in required):
            return False
        # Invalidate if image pca_components changed
        try:
            with open(_pca_img_cached, "rb") as _f:
                _cached_img = pickle.load(_f)
            if _cached_img.n_components_ != pca_components:
                print(f"PCA cache stale: image n_components={_cached_img.n_components_} "
                      f"!= requested {pca_components}")
                return False
        except Exception:
            return False
        # Invalidate if text n_components changed (CV only)
        if text_encoder != "minilm" and _pca_text_cached and os.path.exists(_pca_text_cached):
            try:
                with open(_pca_text_cached, "rb") as _f:
                    _cached_txt = pickle.load(_f)
                if _cached_txt.n_components_ != n_text_pca_components:
                    print(f"PCA cache stale: text n_components={_cached_txt.n_components_} "
                          f"!= requested {n_text_pca_components}")
                    return False
            except Exception:
                return False
        # Cache is stale if either feature file is newer than X_reduced
        x_mtime = os.path.getmtime(_X_reduced_cached)
        return (os.path.getmtime(TEXT_CACHE) <= x_mtime and
                os.path.getmtime(IMAGE_CACHE) <= x_mtime)

    if use_cache and _pca_cache_valid():
        print("PCA cache is valid — skipping PCA subprocess")
        X_reduced_path = _X_reduced_cached
        pca_img_path   = _pca_img_cached
        pca_text_path  = _pca_text_cached
        _write_step("pca_cached")
    else:
        _write_step("pca")
        import tempfile
        pca_out = tempfile.mktemp(suffix=".json", dir="/tmp")
        pca_proc = _sub.run(
            [sys.executable, "/app/services/run_pca.py",
             TEXT_CACHE, IMAGE_CACHE, text_encoder, pca_out,
             str(pca_components), str(n_text_pca_components)],
            capture_output=True, text=True, cwd="/app",
        )
        if pca_proc.returncode != 0:
            if os.path.exists(pca_out):
                os.unlink(pca_out)
            raise RuntimeError(f"PCA subprocess failed:\n{pca_proc.stderr[-3000:]}")
        with open(pca_out) as _f:
            X_reduced_path, pca_img_path, pca_text_path = json.load(_f)
        os.unlink(pca_out)

    X_reduced = np.load(X_reduced_path)
    with open(pca_img_path, "rb") as f:
        pca_img = pickle.load(f)
    pca_text = None
    if pca_text_path and os.path.exists(pca_text_path):
        with open(pca_text_path, "rb") as f:
            pca_text = pickle.load(f)
    pca_models = {"image": pca_img, "text": pca_text}

    y = train_data["prdtypecode"]

    # --- Build and train (TF initializes here, safely isolated from uvicorn heap) ---
    _write_step("training")
    (model, label_encoder, history, model_path, run_id, model_version, eval_results) = (
        build_and_train_model(
            X_reduced, y,
            epochs=epochs, batch_size=batch_size,
            run_name=f"{text_encoder}_train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}",
            pca_models=pca_models,
            train_data=train_data,
            x_csv_path=X_CSV,
            y_csv_path=Y_CSV,
            text_encoder=text_encoder,
            use_dev_images=use_dev_images,
        )
    )

    save_artifacts(model, text_vectorizer, pca_models, label_encoder, skip_existing=False, text_encoder=text_encoder)

    # Notify predict-api to reload artifacts
    predict_api_url = os.getenv("PREDICT_API_URL", "http://predict-api:5003")
    try:
        r = requests.post(f"{predict_api_url}/reload-artifacts", timeout=30)
        if r.ok:
            print(f"predict-api artifacts reloaded successfully")
        else:
            print(f"predict-api reload returned {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"Could not notify predict-api to reload (non-blocking): {e}")

    result = {
        "job_id": job_id,
        "status": "success",
        "mode": "DEV" if use_dev_images else "FULL TRAIN",
        "model_path": model_path,
        "mlflow_run_id": run_id,
        "mlflow_model_version": model_version,
        "train_params": {"epochs": epochs, "batch_size": batch_size},
        "final_metrics": eval_results,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "num_classes": len(label_encoder.classes_),
        "dataset_size": len(train_data),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(result)

except Exception as exc:
    _write({
        "job_id": job_id,
        "status": "failed",
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    sys.exit(1)
