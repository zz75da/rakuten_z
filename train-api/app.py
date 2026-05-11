from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter
from services.data_loader import load_and_merge_data
from services.preprocess_text import extract_text_features
from services.preprocess_image import extract_image_features
from services.pca_reducer import reduce_features
from services.trainer import build_and_train_model, evaluate_model
from services.artifacts import save_artifacts
from utils.logger import get_logger
import requests
import os
import uuid
import pickle
import threading
import numpy as np
import mlflow
from datetime import datetime, timezone
from typing import Dict, Any
import httpx

logger = get_logger()
app = FastAPI(title="Train API", description="Preprocessing + training", version="1.0")

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rakuten_z")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
if MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD:
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow configured with DagsHub: {MLFLOW_TRACKING_URI}")

# --- Prometheus metrics ---
train_loss_gauge = Gauge("train_loss", "Training loss per epoch")
val_loss_gauge = Gauge("val_loss", "Validation loss per epoch")
train_acc_gauge = Gauge("train_accuracy", "Training accuracy per epoch")
val_acc_gauge = Gauge("val_accuracy", "Validation accuracy per epoch")
epochs_counter = Counter("epochs_completed_total", "Total epochs completed")

dataset_size_gauge = Gauge("training_dataset_size", "Number of samples in the training set")
num_classes_gauge = Gauge("model_num_classes", "Number of target classes in the trained model")
final_accuracy_gauge = Gauge("model_final_accuracy", "Final training accuracy of last run")
final_val_accuracy_gauge = Gauge("model_final_val_accuracy", "Final validation accuracy of last run")
final_loss_gauge = Gauge("model_final_loss", "Final training loss of last run")
final_val_loss_gauge = Gauge("model_final_val_loss", "Final validation loss of last run")

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

GATE_API_URL = os.getenv("GATE_API_URL", "http://gate-api:5000")
X_CSV = os.getenv("TRAIN_CSV_X_PATH", "/app/data/X_train_update.csv")
Y_CSV = os.getenv("TRAIN_CSV_Y_PATH", "/app/data/Y_train_CVw08PX.csv")

# --- Request model ---
class TrainRequest(BaseModel):
    use_dev_images: bool = True
    epochs: int = 30
    batch_size: int = 128

# --- In-memory job registry ---
_training_jobs: Dict[str, Dict[str, Any]] = {}




async def verify_jwt_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    
    try:
        # On utilise un client asynchrone httpx pour ne pas bloquer l'event loop
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{GATE_API_URL}/validate-token",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0, # Augmenté légèrement pour la sécurité inter-services
            )
        
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        try:
            return resp.json()
        except Exception:
            raise HTTPException(status_code=502, detail="Gate-api returned non-JSON response")
            
    except httpx.RequestError as exc:
        # Gestion spécifique des erreurs réseau httpx
        raise HTTPException(status_code=503, detail=f"Cannot reach gate-api: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


def _run_training_pipeline(job_id: str, use_dev_images: bool, epochs: int, batch_size: int):
    """Execute the full training pipeline in a background thread."""
    mode = "DEV" if use_dev_images else "FULL TRAIN"
    logger.info(f"[job={job_id}] Training pipeline started (mode={mode})")

    try:
        _training_jobs[job_id]["status"] = "running"

        train_data = load_and_merge_data(use_dev_images=use_dev_images)
        os.makedirs("data", exist_ok=True)

        text_features, text_vectorizer = extract_text_features(train_data)
        np.save("data/text_features.npy", text_features)

        image_features_path = extract_image_features(train_data)
        if not image_features_path:
            raise RuntimeError("Image feature extraction produced no features")

        X_reduced_path, pca_img_path, pca_text_path = reduce_features(
            text_features_path="data/text_features.npy",
            image_features_path="data/image_features.npy",
        )
        X_reduced = np.load(X_reduced_path)
        with open(pca_img_path, "rb") as f:
            pca_img = pickle.load(f)
        with open(pca_text_path, "rb") as f:
            pca_text = pickle.load(f)
        pca_models = {"image": pca_img, "text": pca_text}

        y = train_data["prdtypecode"]
        dataset_size_gauge.set(len(train_data))

        (model, label_encoder, history, model_path, run_id, model_version, eval_results) = (
            build_and_train_model(
                X_reduced, y,
                epochs=epochs, batch_size=batch_size,
                pca_models=pca_models,
                train_data=train_data,
                x_csv_path=X_CSV,
                y_csv_path=Y_CSV,
            )
        )

        # Always overwrite so predict-api picks up the freshest artifacts
        save_artifacts(model, text_vectorizer, pca_models, label_encoder, skip_existing=False)

        # Notify predict-api to reload artifacts from disk
        predict_api_url = os.getenv("PREDICT_API_URL", "http://predict-api:5003")
        try:
            r = requests.post(f"{predict_api_url}/reload-artifacts", timeout=30)
            if r.ok:
                logger.info("predict-api artifacts reloaded successfully")
            else:
                logger.warning(f"predict-api reload returned {r.status_code}: {r.text[:200]}")
        except Exception as reload_exc:
            logger.warning(f"Could not notify predict-api to reload (non-blocking): {reload_exc}")

        # Update Prometheus
        num_classes_gauge.set(len(label_encoder.classes_))
        for t_loss, v_loss, t_acc, v_acc in zip(
            history.history.get("loss", []),
            history.history.get("val_loss", []),
            history.history.get("accuracy", []),
            history.history.get("val_accuracy", []),
        ):
            train_loss_gauge.set(t_loss)
            val_loss_gauge.set(v_loss)
            train_acc_gauge.set(t_acc)
            val_acc_gauge.set(v_acc)
            epochs_counter.inc()

        losses = history.history.get("loss", [])
        val_losses = history.history.get("val_loss", [])
        accs = history.history.get("accuracy", [])
        val_accs = history.history.get("val_accuracy", [])
        if losses:
            final_loss_gauge.set(losses[-1])
        if val_losses:
            final_val_loss_gauge.set(val_losses[-1])
        if accs:
            final_accuracy_gauge.set(accs[-1])
        if val_accs:
            final_val_accuracy_gauge.set(val_accs[-1])

        result = {
            "status": "success",
            "mode": mode,
            "model_path": model_path,
            "mlflow_run_id": run_id,
            "mlflow_model_version": model_version,
            "train_params": {"epochs": epochs, "batch_size": batch_size},
            "final_metrics": eval_results,
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _training_jobs[job_id].update(result)
        logger.info(f"[job={job_id}] Training completed successfully")

    except Exception as exc:
        logger.exception(f"[job={job_id}] Training failed: {exc}")
        _training_jobs[job_id].update({
            "status": "failed",
            "error": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


# --- Endpoints ---

@app.post("/train", status_code=202)
async def train_model(
    req: TrainRequest,
    user: dict = Depends(verify_jwt_token),
):
    """Start training asynchronously. Returns a job_id to poll with GET /train/status/{job_id}."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admin can trigger training")

    job_id = str(uuid.uuid4())
    _training_jobs[job_id] = {
        "status": "running",
        "job_id": job_id,
        "started_by": user.get("username"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    thread = threading.Thread(
        target=_run_training_pipeline,
        args=(job_id, req.use_dev_images, req.epochs, req.batch_size),
        daemon=True,
    )
    thread.start()

    logger.info(f"Training job {job_id} started by {user.get('username')}")
    return {
        "job_id": job_id,
        "status": "running",
        "message": f"Training started — poll /train/status/{job_id}",
    }


@app.get("/train/status/{job_id}")
async def training_status(job_id: str, user: dict = Depends(verify_jwt_token)):
    """Poll training job status. Status: running | success | failed."""
    job = _training_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.get("/health")
def health():
    running = sum(1 for j in _training_jobs.values() if j.get("status") == "running")
    return {"status": "healthy", "service": "train-api", "version": "1.0", "active_jobs": running}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
