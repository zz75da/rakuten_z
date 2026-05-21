from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter
from utils.logger import get_logger
import os
import sys
import uuid
import json
import threading
import subprocess
import httpx
from datetime import datetime, timezone
from typing import Dict, Any

logger = get_logger()
app = FastAPI(title="Train API", description="Preprocessing + training", version="1.0")

# --- Prometheus metrics ---
train_loss_gauge = Gauge("train_loss", "Training loss per epoch")
val_loss_gauge = Gauge("val_loss", "Validation loss per epoch")
train_acc_gauge = Gauge("train_accuracy", "Training accuracy per epoch")
val_acc_gauge = Gauge("val_accuracy", "Validation accuracy per epoch")
epochs_counter = Counter("epochs_completed_total", "Total epochs completed")
last_run_epochs_gauge = Gauge("model_last_run_epochs", "Epochs trained in the last completed run")

dataset_size_gauge = Gauge("training_dataset_size", "Number of samples in the training set")
num_classes_gauge = Gauge("model_num_classes", "Number of target classes in the trained model")
final_accuracy_gauge = Gauge("model_final_accuracy", "Final training accuracy of last run", ["encoder"])
final_val_accuracy_gauge = Gauge("model_final_val_accuracy", "Final validation accuracy of last run", ["encoder"])
final_loss_gauge = Gauge("model_final_loss", "Final training loss of last run", ["encoder"])
final_val_loss_gauge = Gauge("model_final_val_loss", "Final validation loss of last run", ["encoder"])

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

GATE_API_URL = os.getenv("GATE_API_URL", "http://gate-api:5000")
X_CSV = os.getenv("TRAIN_CSV_X_PATH", "/app/data/X_train_update.csv")
Y_CSV = os.getenv("TRAIN_CSV_Y_PATH", "/app/data/Y_train_CVw08PX.csv")
JOB_DIR = "/app/data/jobs"

# --- Request model ---
class TrainRequest(BaseModel):
    use_dev_images: bool = False
    epochs: int = 30
    batch_size: int = 128
    use_cache: bool = True
    text_encoder: str = "countvectorizer"  # "countvectorizer" | "minilm" | "clip"

# --- In-memory job registry ---
_training_jobs: Dict[str, Dict[str, Any]] = {}


def _load_persisted_jobs():
    """Recover job results from disk on startup — prevents 404s after restarts."""
    if not os.path.exists(JOB_DIR):
        return
    for fname in os.listdir(JOB_DIR):
        if not fname.endswith(".json"):
            continue
        job_id = fname[:-5]
        try:
            with open(os.path.join(JOB_DIR, fname)) as f:
                job = json.load(f)
            if job.get("status") == "running":
                job["status"] = "interrupted"
                job["error"] = "train-api restarted while job was running — re-trigger a new DAG run"
            _training_jobs[job_id] = job
        except Exception as e:
            logger.warning(f"Could not load persisted job {job_id}: {e}")


def _restore_metrics_from_jobs():
    """Restore Prometheus Gauges from the most recent successful job files.

    Gauges reset to 0 on every restart. Scanning job files lets Grafana show
    real values immediately without waiting for the next training run.
    """
    if not os.path.isdir(JOB_DIR):
        return
    best: Dict[str, Any] = {}  # encoder -> most recent successful job
    for fname in os.listdir(JOB_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(JOB_DIR, fname)) as f:
                job = json.load(f)
            if job.get("status") != "success":
                continue
            _mp = job.get("model_path", "")
            enc = "minilm" if "_minilm" in _mp else "clip" if "_clip" in _mp else "countvectorizer"
            if enc not in best or job.get("completed_at", "") > best[enc].get("completed_at", ""):
                best[enc] = job
        except Exception:
            continue

    if not best:
        return

    # Shared metrics: use the most recent job across all encoders
    most_recent = max(best.values(), key=lambda j: j.get("completed_at", ""))
    num_classes_gauge.set(most_recent.get("num_classes", 0))
    dataset_size_gauge.set(most_recent.get("dataset_size", 0))

    # Per-encoder metrics
    for enc, job in best.items():
        history = job.get("history", {})
        losses    = history.get("loss", [])
        val_losses = history.get("val_loss", [])
        accs      = history.get("accuracy", [])
        val_accs  = history.get("val_accuracy", [])
        if losses:
            final_loss_gauge.labels(encoder=enc).set(losses[-1])
        if val_losses:
            final_val_loss_gauge.labels(encoder=enc).set(val_losses[-1])
        if accs:
            final_accuracy_gauge.labels(encoder=enc).set(accs[-1])
        if val_accs:
            final_val_accuracy_gauge.labels(encoder=enc).set(val_accs[-1])
        last_run_epochs_gauge.set(len(losses))

    logger.info(f"Prometheus metrics restored from job files: encoders={list(best.keys())}")


_load_persisted_jobs()
_restore_metrics_from_jobs()


async def verify_jwt_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{GATE_API_URL}/validate-token",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )

        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        try:
            return resp.json()
        except Exception:
            raise HTTPException(status_code=502, detail="Gate-api returned non-JSON response")

    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Cannot reach gate-api: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


def _run_training_pipeline(job_id: str, use_dev_images: bool, epochs: int, batch_size: int, use_cache: bool = True, text_encoder: str = "countvectorizer"):
    """Spawn the full training pipeline in an isolated subprocess.

    TF never initializes in the main uvicorn process — eliminates the
    'double free or corruption (fasttop)' crash caused by TF's allocator
    hooks conflicting with glibc's heap after PCA frees large arrays.
    """
    mode = "DEV" if use_dev_images else "FULL TRAIN"
    logger.info(f"[job={job_id}] Spawning training subprocess (mode={mode})")

    os.makedirs(JOB_DIR, exist_ok=True)

    try:
        _training_jobs[job_id]["status"] = "running"

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(
            [
                sys.executable, "/app/services/run_full_pipeline.py",
                job_id, text_encoder,
                str(epochs), str(batch_size),
                str(use_dev_images), str(use_cache),
            ],
            stdout=None, stderr=None, cwd="/app", env=env,
        )

        # Read result written by subprocess
        job_file = os.path.join(JOB_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            with open(job_file) as f:
                result = json.load(f)
            _training_jobs[job_id].update(result)

        if proc.returncode != 0:
            if _training_jobs[job_id].get("status") != "failed":
                _training_jobs[job_id].update({
                    "status": "failed",
                    "error": f"subprocess rc={proc.returncode} — see docker logs train-api for details",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
            logger.error(f"[job={job_id}] Training subprocess failed (rc={proc.returncode}) — see docker logs for details")
            return

        logger.info(f"[job={job_id}] Training completed successfully")

        # Update Prometheus from subprocess result
        result = _training_jobs[job_id]
        num_classes_gauge.set(result.get("num_classes", 0))
        dataset_size_gauge.set(result.get("dataset_size", 0))
        history_data = result.get("history", {})
        epoch_losses = history_data.get("loss", [])
        for t_loss, v_loss, t_acc, v_acc in zip(
            epoch_losses,
            history_data.get("val_loss", []),
            history_data.get("accuracy", []),
            history_data.get("val_accuracy", []),
        ):
            train_loss_gauge.set(t_loss)
            val_loss_gauge.set(v_loss)
            train_acc_gauge.set(t_acc)
            val_acc_gauge.set(v_acc)
            epochs_counter.inc()
        last_run_epochs_gauge.set(len(epoch_losses))
        losses = history_data.get("loss", [])
        val_losses = history_data.get("val_loss", [])
        accs = history_data.get("accuracy", [])
        val_accs = history_data.get("val_accuracy", [])
        enc = text_encoder
        if losses:
            final_loss_gauge.labels(encoder=enc).set(losses[-1])
        if val_losses:
            final_val_loss_gauge.labels(encoder=enc).set(val_losses[-1])
        if accs:
            final_accuracy_gauge.labels(encoder=enc).set(accs[-1])
        if val_accs:
            final_val_accuracy_gauge.labels(encoder=enc).set(val_accs[-1])

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

    # Reject if a job is already running — all jobs share the same feature cache paths
    running = [j for j in _training_jobs.values() if j.get("status") == "running"]
    if running:
        raise HTTPException(
            status_code=409,
            detail=f"Training job {running[0]['job_id']} already in progress — wait for it to finish",
        )

    job_id = str(uuid.uuid4())
    _training_jobs[job_id] = {
        "status": "running",
        "job_id": job_id,
        "started_by": user.get("username"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    thread = threading.Thread(
        target=_run_training_pipeline,
        args=(job_id, req.use_dev_images, req.epochs, req.batch_size, req.use_cache, req.text_encoder),
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
async def training_status(job_id: str, _user: dict = Depends(verify_jwt_token)):
    """Poll training job status. Status: running | success | failed | interrupted."""
    job = _training_jobs.get(job_id)
    if job is None:
        # Fallback: check persisted file (survives restarts)
        job_file = os.path.join(JOB_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            with open(job_file) as f:
                job = json.load(f)
            _training_jobs[job_id] = job
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.post("/push-cache")
async def push_cache_endpoint():
    """
    Run `dvc add -f` on all existing .npy feature cache files then `dvc push`.
    Called automatically by the Airflow DAG after both training runs complete.
    """
    cache_dir = "/app/data/feature_cache"
    npy_files = [
        "image_features.npy",
        "text_features.npy",
        "text_features_minilm.npy",
        "text_features_clip.npy",
    ]

    results = {}

    for fname in npy_files:
        fpath = os.path.join(cache_dir, fname)
        if os.path.exists(fpath):
            r = subprocess.run(
                ["dvc", "add", "-f", fpath],
                capture_output=True, text=True, cwd="/app",
            )
            results[f"add_{fname}"] = {
                "rc": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()
            }
            if r.returncode != 0:
                logger.warning(f"dvc add {fname} failed: {r.stderr.strip()}")
            else:
                logger.info(f"dvc add {fname}: OK")
        else:
            results[f"add_{fname}"] = {"rc": -1, "stdout": "", "stderr": "file not found — skipped"}

    push = subprocess.run(
        ["dvc", "push"],
        capture_output=True, text=True, cwd="/app",
    )
    results["push"] = {
        "rc": push.returncode, "stdout": push.stdout.strip(), "stderr": push.stderr.strip()
    }
    if push.returncode != 0:
        logger.error(f"dvc push failed: {push.stderr.strip()}")
        raise HTTPException(status_code=500, detail=f"dvc push failed: {push.stderr.strip()[:500]}")

    logger.info("DVC feature cache pushed to remote successfully")
    return {"status": "ok", "details": results}


@app.get("/health")
def health():
    running = sum(1 for j in _training_jobs.values() if j.get("status") == "running")
    return {"status": "healthy", "service": "train-api", "version": "1.0", "active_jobs": running}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
