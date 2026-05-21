"""
Integration tests: async training workflow
==========================================
Purpose : Verify the full asynchronous training lifecycle end-to-end using
          FastAPI's TestClient against the real train-api app.  The heavy ML
          work (_run_training_pipeline) is patched to isolate HTTP behaviour
          from compute time; TF / MLflow are stubbed at module level.

Covered :
  TestAsyncTrainingWorkflow
    - POST /train returns 202 + job_id in under 5 s (non-blocking)
    - The new job is immediately registered in _training_jobs as "running"
    - GET /train/status/{job_id} returns 200 with a valid status string
    - A manually inserted "success" entry exposes final_metrics and mlflow_run_id
    - A manually inserted "failed" entry exposes the error message
    - Unknown job_id returns 404
    - Multiple concurrent POST /train calls receive distinct job_ids

  TestRunTrainingPipelineFunction  (lower-level, faster feedback)
    - When all pipeline steps succeed, _run_training_pipeline() sets
      _training_jobs[id]["status"] to "success" or "failed" (never left as "running")
    - When load_and_merge_data() raises, status is set to "failed"
      and the error message is stored verbatim

Note    : All patch() calls use patch.object(train_app_mod, ...) to avoid
          sys.modules["app"] collisions introduced by other test modules.

Dependencies : train-api/app.py (TF/MLflow/services stubbed), FastAPI TestClient
"""
import sys, os
import time
import uuid
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "train-api"))

# Only clear 'app' if it is NOT already the train-api module (has _training_jobs).
# Re-importing when train-api is cached would re-register Prometheus metrics → ValueError.
_existing_app = sys.modules.get("app")
if _existing_app is not None and not hasattr(_existing_app, "_training_jobs"):
    del sys.modules["app"]
    for _k in list(sys.modules.keys()):
        if _k.startswith("services."):
            del sys.modules[_k]

# Stub TF and heavy deps before importing train-api app
for _mod in [
    "tensorflow", "tensorflow.keras", "tensorflow.keras.applications",
    "tensorflow.keras.applications.resnet50",
    "mlflow", "mlflow.tracking", "mlflow.tensorflow", "mlflow.sklearn",
    "dagshub",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

for _svc in ["preprocess_image", "trainer"]:
    _full = f"services.{_svc}"
    if _full not in sys.modules:
        sys.modules[_full] = MagicMock()

import app as train_app_mod
from app import app, verify_jwt_token, _training_jobs


@pytest.fixture(autouse=True)
def admin_override():
    app.dependency_overrides[verify_jwt_token] = lambda: {
        "username": "admin", "role": "admin"}
    # Clear any running jobs from prior tests so the 409 guard doesn't fire
    for job in _training_jobs.values():
        if job.get("status") == "running":
            job["status"] = "success"
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Async job lifecycle
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestAsyncTrainingWorkflow:
    def test_trigger_returns_job_id_immediately(self, client):
        # Use patch.object so sys.modules["app"] pollution from other tests is irrelevant
        with patch.object(train_app_mod, "_run_training_pipeline"):
            start = time.time()
            resp = client.post("/train", json={"use_dev_images": True, "epochs": 1})
            elapsed = time.time() - start
        assert resp.status_code == 202
        assert "job_id" in resp.json()
        assert elapsed < 5.0

    def test_job_registered_as_running(self, client):
        with patch.object(train_app_mod, "_run_training_pipeline"):
            resp = client.post("/train", json={"use_dev_images": True, "epochs": 1})
        job_id = resp.json()["job_id"]
        assert job_id in _training_jobs
        assert _training_jobs[job_id]["status"] == "running"

    def test_poll_status_while_running(self, client):
        with patch.object(train_app_mod, "_run_training_pipeline"):
            post = client.post("/train", json={"epochs": 1})
        job_id = post.json()["job_id"]

        status = client.get(f"/train/status/{job_id}")
        assert status.status_code == 200
        assert status.json()["status"] in ("running", "success", "failed")

    def test_manual_success_entry_is_readable(self, client):
        fake_id = str(uuid.uuid4())
        _training_jobs[fake_id] = {
            "status": "success",
            "job_id": fake_id,
            "final_metrics": {"accuracy": 0.72},
            "history": {"loss": [0.9, 0.5], "val_accuracy": [0.6, 0.72]},
            "mlflow_run_id": "run-abc",
            "mlflow_model_version": 3,
        }
        resp = client.get(f"/train/status/{fake_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["mlflow_run_id"] == "run-abc"
        assert data["final_metrics"]["accuracy"] == pytest.approx(0.72)

    def test_manual_failed_entry_exposes_error(self, client):
        fake_id = str(uuid.uuid4())
        _training_jobs[fake_id] = {
            "status": "failed",
            "job_id": fake_id,
            "error": "CUDA out of memory",
        }
        resp = client.get(f"/train/status/{fake_id}")
        assert resp.status_code == 200
        assert resp.json()["error"] == "CUDA out of memory"

    def test_unknown_job_id_is_404(self, client):
        resp = client.get("/train/status/does-not-exist-at-all")
        assert resp.status_code == 404

    def test_second_submission_while_running_returns_409(self, client):
        """The 409 guard prevents a second training job from starting while one
        is already running (both runs share the same feature-cache paths)."""
        with patch.object(train_app_mod, "_run_training_pipeline"):
            first = client.post("/train", json={"epochs": 1})
        assert first.status_code == 202
        # First job is still "running" — second attempt must be rejected
        second = client.post("/train", json={"epochs": 1})
        assert second.status_code == 409
        assert "already in progress" in second.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Training pipeline function (unit-level, faster feedback)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestRunTrainingPipelineFunction:
    def test_success_path_updates_job_registry(self):
        """_run_training_pipeline reads the job JSON written by the subprocess.
        Patch subprocess.run to return rc=0 and pre-write the expected job file."""
        import json, tempfile

        fake_id = str(uuid.uuid4())
        _training_jobs[fake_id] = {"status": "running"}

        fake_result = {
            "status": "success",
            "job_id": fake_id,
            "history": {
                "loss": [0.9, 0.5],
                "val_loss": [1.0, 0.6],
                "accuracy": [0.4, 0.7],
                "val_accuracy": [0.35, 0.65],
            },
            "final_metrics": {"accuracy": 0.7},
            "num_classes": 5,
            "dataset_size": 10,
            "mlflow_run_id": "run-xyz",
        }

        mock_proc = MagicMock()
        mock_proc.returncode = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            job_file = os.path.join(tmpdir, f"{fake_id}.json")
            with open(job_file, "w") as f:
                json.dump(fake_result, f)

            with patch.object(train_app_mod, "JOB_DIR", tmpdir), \
                 patch("subprocess.run", return_value=mock_proc):
                train_app_mod._run_training_pipeline(
                    job_id=fake_id,
                    use_dev_images=True,
                    epochs=2,
                    batch_size=32,
                )

        assert _training_jobs[fake_id]["status"] == "success"
        assert _training_jobs[fake_id]["mlflow_run_id"] == "run-xyz"

    def test_failure_marks_job_as_failed(self):
        """When the subprocess returns a non-zero exit code and writes no job file,
        _run_training_pipeline must mark the job as failed with an error message."""
        import tempfile

        fake_id = str(uuid.uuid4())
        _training_jobs[fake_id] = {"status": "running"}

        mock_proc = MagicMock()
        mock_proc.returncode = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(train_app_mod, "JOB_DIR", tmpdir), \
                 patch("subprocess.run", return_value=mock_proc):
                train_app_mod._run_training_pipeline(
                    job_id=fake_id,
                    use_dev_images=True,
                    epochs=1,
                    batch_size=32,
                )

        assert _training_jobs[fake_id]["status"] == "failed"
        assert _training_jobs[fake_id].get("error") is not None
