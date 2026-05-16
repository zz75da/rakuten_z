"""
Unit tests for predict-api endpoints
======================================
Purpose : Validate all prediction endpoints in isolation — no real model,
          no gate-api call, no disk I/O.  TensorFlow, MLflow, and cv2 are
          stubbed in sys.modules; ML artefacts are replaced with MagicMocks
          injected via the autouse inject_mock_artifacts fixture.

Covered :
  TestHealth           — GET /health returns status="ok"

  TestPredictText      — POST /predict-text
    - Returns 200 with pred_class, label, probs, mode="text_only"
    - Probabilities sum to 1.0 (softmax invariant)
    - Missing description field returns 422
    - Empty description is handled gracefully (200 or 400)

  TestPredictImage     — POST /predict-image
    - Valid base64 JPEG returns 200 with mode="image_only"
    - Truly invalid base64 (cv2 returns None) returns 400
    - Missing image_base64 field returns 422

  TestPredictMultimodal — POST /predict-multimodal
    - Both modalities → 200, mode="multimodal"
    - Text-only or image-only → 200
    - Neither field → 400

  TestReloadArtifacts  — POST /reload-artifacts
    - Successful reload returns status="reloaded" + PCA component counts
    - RuntimeError from load_artifacts propagates as 500

  TestDriftMetrics
    - PREDICTION_CONFIDENCE, PREDICTION_ENTROPY, PREDICTION_CLASS_COUNT
      Prometheus metrics are observed on each prediction call

Note    : All patch() calls use patch.object(predict_app, ...) to avoid
          sys.modules["app"] pollution from other service test modules.

Dependencies : predict-api/app.py (TF/MLflow/cv2 mocked), conftest mock fixtures
"""
import sys, os
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "predict-api"))

# Clear any previously cached 'app' module (gate-api or train-api may have been imported first)
for _k in list(sys.modules.keys()):
    if _k == "app":
        del sys.modules[_k]

# Stub heavy optional dependencies before importing app
for _mod in [
    "tensorflow", "tensorflow.keras", "tensorflow.keras.applications",
    "tensorflow.keras.applications.resnet50", "tensorflow.keras.preprocessing",
    "tensorflow.keras.preprocessing.image",
    "mlflow", "mlflow.tracking", "mlflow.tensorflow", "mlflow.sklearn",
    "cv2", "dagshub",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Provide a realistic tf.keras mock so `import tensorflow as tf; tf.keras...` works
_tf_mock = sys.modules["tensorflow"]
_tf_mock.keras = sys.modules["tensorflow.keras"]


# ---------------------------------------------------------------------------
# Import app with startup suppressed so we control artefacts
# patch("app.load_artifacts") is safe here because sys.modules["app"] is still
# predict-api at collection time (cleared above). At runtime we use patch.object.
# ---------------------------------------------------------------------------
with patch("app.load_artifacts"):
    import app as predict_app
    from app import app as _predict_fastapi_app, verify_jwt_token

# Keep a stable alias so fixture/test code never resolves through sys.modules["app"]
app = _predict_fastapi_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def inject_mock_artifacts(mock_model, mock_label_encoder, mock_pca_image,
                          mock_pca_text, mock_vectorizer, mock_minilm_encoder):
    """Inject mock ML objects into predict-api module globals.

    predict-api now has separate model_cv / model_minilm globals (dual-model
    architecture).  model_minilm and minilm_encoder are set to None by default
    so CV-only tests don't accidentally resolve the MiniLM branch.
    resnet_model is mocked so extract_image_features() doesn't hit real ResNet.
    """
    mock_resnet = MagicMock()
    mock_resnet.predict.return_value = np.zeros((1, 2048), dtype=np.float32)

    predict_app.model_cv        = mock_model
    predict_app.model_minilm    = None
    predict_app.minilm_encoder  = None
    predict_app.label_encoder   = mock_label_encoder
    predict_app.pca_image       = mock_pca_image
    predict_app.pca_text        = mock_pca_text
    predict_app.text_vectorizer = mock_vectorizer
    predict_app.resnet_model    = mock_resnet
    yield


@pytest.fixture
def client(valid_admin_token):
    """TestClient with gate-api auth bypassed via dependency override."""
    fake_user = {"username": "admin", "role": "admin", "token": valid_admin_token}

    def _override():
        return fake_user

    app.dependency_overrides[verify_jwt_token] = _override
    # Use patch.object(predict_app, ...) so the patch targets the predict-api module
    # even when sys.modules["app"] points to another service's module.
    with patch.object(predict_app, "load_artifacts"):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_ok(self):
        with patch.object(predict_app, "load_artifacts"):
            with TestClient(app) as c:
                resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /predict-text
# ---------------------------------------------------------------------------
class TestPredictText:
    def test_returns_200_with_correct_keys(self, client):
        resp = client.post("/predict-text",
                           json={"description": "leather handbag high quality"})
        assert resp.status_code == 200
        data = resp.json()
        assert "pred_class" in data
        assert "label" in data
        assert "probs" in data
        assert "encoder" in data
        assert data["mode"] == "text_only"
        assert data["encoder"] == "cv"  # default encoder

    def test_encoder_field_reflects_selection(self, client):
        resp = client.post("/predict-text",
                           json={"description": "shoes", "model": "cv"})
        assert resp.status_code == 200
        assert resp.json()["encoder"] == "cv"

    def test_minilm_unavailable_returns_503(self, client):
        # model_minilm is None (set by inject_mock_artifacts) → 503
        resp = client.post("/predict-text",
                           json={"description": "shoes", "model": "minilm"})
        assert resp.status_code == 503

    def test_probs_sum_to_one(self, client):
        predict_app.model_cv.predict.return_value = np.array([[0.2, 0.5, 0.1, 0.1, 0.1]])
        resp = client.post("/predict-text", json={"description": "shoes"})
        assert resp.status_code == 200
        probs = resp.json()["probs"][0]
        assert abs(sum(probs) - 1.0) < 1e-3

    def test_no_description_is_422(self, client):
        resp = client.post("/predict-text", json={})
        assert resp.status_code == 422

    def test_empty_description_handled(self, client):
        resp = client.post("/predict-text", json={"description": ""})
        # Empty string still calls the pipeline — 200 or 400 are both acceptable
        assert resp.status_code in (200, 400)


# ---------------------------------------------------------------------------
# /predict-image
# ---------------------------------------------------------------------------
class TestPredictImage:
    def test_valid_image_returns_200(self, client, sample_image_b64):
        resp = client.post("/predict-image",
                           json={"image_base64": sample_image_b64})
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "image_only"
        assert "pred_class" in data

    def test_invalid_base64_returns_400(self, client):
        # cv2 is mocked, so we must force imdecode to return None to simulate failure
        import cv2 as _cv2_mock
        _cv2_mock.imdecode.return_value = None
        resp = client.post("/predict-image",
                           json={"image_base64": "this-is-not-valid-base64!!"})
        _cv2_mock.imdecode.return_value = MagicMock()  # restore default
        assert resp.status_code == 400

    def test_missing_field_is_422(self, client):
        resp = client.post("/predict-image", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /predict-multimodal
# ---------------------------------------------------------------------------
class TestPredictMultimodal:
    def test_both_modalities(self, client, sample_image_b64):
        resp = client.post("/predict-multimodal", json={
            "description": "leather bag",
            "image_base64": sample_image_b64,
        })
        assert resp.status_code == 200
        assert resp.json()["mode"] == "multimodal"

    def test_text_only_multimodal(self, client):
        resp = client.post("/predict-multimodal",
                           json={"description": "running shoes"})
        assert resp.status_code == 200

    def test_image_only_multimodal(self, client, sample_image_b64):
        resp = client.post("/predict-multimodal",
                           json={"image_base64": sample_image_b64})
        assert resp.status_code == 200

    def test_neither_field_is_400(self, client):
        resp = client.post("/predict-multimodal", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /reload-artifacts
# ---------------------------------------------------------------------------
class TestReloadArtifacts:
    def test_reload_success(self, client):
        with patch.object(predict_app, "load_artifacts"):
            resp = client.post("/reload-artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reloaded"
        assert "cv_model_loaded" in data
        assert "minilm_model_loaded" in data
        assert "pca_image_components" in data
        assert data["pca_image_components"] == 300

    def test_reload_failure_returns_500(self, client):
        with patch.object(predict_app, "load_artifacts", side_effect=RuntimeError("disk error")):
            resp = client.post("/reload-artifacts")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Drift metrics are recorded (no exception = pass)
# ---------------------------------------------------------------------------
class TestDriftMetrics:
    def test_confidence_metric_observed(self, client):
        with patch.object(predict_app, "PREDICTION_CONFIDENCE") as mock_conf:
            client.post("/predict-text", json={"description": "test"})
            mock_conf.labels.assert_called()

    def test_entropy_metric_observed(self, client):
        with patch.object(predict_app, "PREDICTION_ENTROPY") as mock_ent:
            client.post("/predict-text", json={"description": "test"})
            mock_ent.labels.assert_called()

    def test_class_counter_incremented(self, client):
        with patch.object(predict_app, "PREDICTION_CLASS_COUNT") as mock_cls:
            client.post("/predict-text", json={"description": "test"})
            mock_cls.labels.assert_called()
