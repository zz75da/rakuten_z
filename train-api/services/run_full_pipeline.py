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

job_id          = sys.argv[1]
text_encoder    = sys.argv[2]
epochs          = int(sys.argv[3])
batch_size      = int(sys.argv[4])
use_dev_images  = sys.argv[5].lower() == "true"
use_cache       = sys.argv[6].lower() == "true"
git_commit_sha  = sys.argv[7] if len(sys.argv) > 7 else ""


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
                        "pca_components":       int(pre.get("pca_components", 384)),
                        "n_text_pca_components": int(pre.get("n_text_pca_components", 512)),
                        "cv_max_features":       int(pre.get("cv_max_features", 10000)),
                    }
                except Exception:
                    pass
        return {"pca_components": 384, "n_text_pca_components": 512, "cv_max_features": 10000}

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
                "MiniLM feature cache not found — trigger the DAG to run minilm encoding first."
            )
        text_vectorizer = None
    elif text_encoder == "mpnet":
        TEXT_CACHE = "/app/data/feature_cache/text_features_mpnet.npy"
        if not os.path.exists(TEXT_CACHE):
            raise RuntimeError(
                "mpnet feature cache not found — trigger the DAG to run minilm/mpnet encoding first."
            )
        text_vectorizer = None
    elif text_encoder == "clip":
        TEXT_CACHE = "/app/data/feature_cache/text_features_clip.npy"
        if not os.path.exists(TEXT_CACHE):
            raise RuntimeError(
                "CLIP feature cache not found — trigger the DAG to run clip encoding first."
            )
        text_vectorizer = None
    else:
        TEXT_CACHE = "/app/data/feature_cache/text_features.npy"
        VECTORIZER_CACHE = "/app/data/artifacts/text_vectorizer.pkl"
        from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
        _OCR_CACHE      = "/app/data/feature_cache/ocr_text.csv"
        _TEXT_META      = "/app/data/feature_cache/text_features_meta.json"

        def _text_cache_valid():
            if not (os.path.exists(TEXT_CACHE) and os.path.exists(VECTORIZER_CACHE)):
                return False
            try:
                with open(VECTORIZER_CACHE, "rb") as _f:
                    _vec = pickle.load(_f)
                # Reject stale cache if vectorizer class changed (e.g. CountVectorizer → TfidfVectorizer)
                if not isinstance(_vec, _TfidfVectorizer):
                    print(f"Text cache stale: cached {type(_vec).__name__} but code uses TfidfVectorizer")
                    return False
                cached_vocab = len(_vec.vocabulary_)
                if cached_vocab != cv_max_features:
                    print(f"Text cache stale: vocab={cached_vocab} != requested {cv_max_features}")
                    return False
                # Reject stale cache if OCR availability changed since cache was built
                ocr_now = os.path.exists(_OCR_CACHE)
                if os.path.exists(_TEXT_META):
                    with open(_TEXT_META) as _mf:
                        meta = json.load(_mf)
                    if meta.get("ocr_used") != ocr_now:
                        print(f"Text cache stale: ocr_used changed "
                              f"({meta.get('ocr_used')} → {ocr_now})")
                        return False
                elif ocr_now:
                    # OCR cache exists but text was built without it
                    print("Text cache stale: OCR cache found but text was cached without OCR")
                    return False
                return True
            except Exception:
                return False

        def _write_text_meta():
            with open(_TEXT_META, "w") as _mf:
                json.dump({"ocr_used": os.path.exists(_OCR_CACHE)}, _mf)

        if use_cache and _text_cache_valid():
            with open(VECTORIZER_CACHE, "rb") as f:
                text_vectorizer = pickle.load(f)
        elif (use_cache
              and os.path.exists(TEXT_CACHE)
              and not os.path.exists(VECTORIZER_CACHE)):
            # text_features.npy is valid but vectorizer was deleted/corrupted.
            # Re-fit vectorizer only (fit_only=True) — avoids the 3.4 GB toarray() OOM.
            print("text_features.npy present but vectorizer missing — re-fitting vectorizer only")
            _, text_vectorizer = extract_text_features(
                train_data, max_features=cv_max_features, fit_only=True
            )
            with open(VECTORIZER_CACHE, "wb") as f:
                pickle.dump(text_vectorizer, f)
            _write_text_meta()
        else:
            text_features, text_vectorizer = extract_text_features(
                train_data, max_features=cv_max_features
            )
            np.save(TEXT_CACHE, text_features)
            _write_text_meta()   # record whether OCR was used so cache stays consistent

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
    # Cache check: compare a lightweight fingerprint (encoder + PCA params + feature
    # file sizes) stored in a JSON sidecar.  File sizes change only when features are
    # actually regenerated; mtimes are NOT used because DVC add/push and np.save can
    # update them even when the content is identical — which caused spurious PCA reruns.
    _ARTIFACTS = "/app/data/artifacts"
    # Versioned paths: encoder + pca_components in filename so different PCA configs
    # coexist on disk. Switching pca_components hits cache instead of recomputing 6h.
    _X_reduced_cached = os.path.join(_ARTIFACTS, f"X_reduced_{text_encoder}_{pca_components}.npy")
    _pca_img_cached   = os.path.join(_ARTIFACTS, f"pca_image_{pca_components}.pkl")
    _pca_text_cached  = (os.path.join(_ARTIFACTS, f"pca_text_{n_text_pca_components}.pkl")
                         if text_encoder not in ("minilm", "clip", "mpnet") else None)
    _pca_fingerprint  = os.path.join(_ARTIFACTS, f"pca_fingerprint_{text_encoder}_{pca_components}.json")

    def _make_pca_fingerprint() -> dict:
        """Cheap fingerprint: encoder + PCA params + feature file sizes."""
        return {
            "text_encoder":          text_encoder,
            "pca_components":        pca_components,
            "n_text_pca_components": n_text_pca_components,
            "text_size":             os.path.getsize(TEXT_CACHE),
            "image_size":            os.path.getsize(IMAGE_CACHE),
        }

    def _pca_cache_valid():
        required = [_X_reduced_cached, _pca_img_cached, _pca_fingerprint]
        if text_encoder not in ("minilm", "clip", "mpnet"):
            required.append(_pca_text_cached)
        if not all(os.path.exists(p) for p in required):
            return False
        # Compare stored fingerprint with current inputs
        try:
            with open(_pca_fingerprint) as _f:
                stored = json.load(_f)
            if stored != _make_pca_fingerprint():
                print(f"PCA fingerprint mismatch: {stored} != {_make_pca_fingerprint()}")
                return False
        except Exception as _e:
            print(f"PCA fingerprint read failed: {_e}")
            return False
        # Sanity-check PCA pkl n_components match params
        try:
            with open(_pca_img_cached, "rb") as _f:
                _cached_img = pickle.load(_f)
            if _cached_img.n_components_ != pca_components:
                print(f"PCA cache stale: image n_components={_cached_img.n_components_} "
                      f"!= requested {pca_components}")
                return False
        except Exception:
            return False
        return True

    def _write_pca_fingerprint():
        with open(_pca_fingerprint, "w") as _f:
            json.dump(_make_pca_fingerprint(), _f)
        print(f"PCA fingerprint written -> {_pca_fingerprint}")

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
        _write_pca_fingerprint()  # stamp fingerprint so next run skips PCA

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
            git_commit_sha=git_commit_sha,
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
