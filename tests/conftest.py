"""
Shared fixtures and session-level plugins for the Rakuten MLOps test suite
===========================================================================
Purpose : Provide reusable test infrastructure for all unit and integration
          tests without duplicating setup code across test files.

Fixtures provided :
  artifacts_module       — lazily imported train-api artifacts.py module
  pca_reducer_module     — lazily imported train-api pca_reducer.py module
  preprocess_text_module — lazily imported train-api preprocess_text.py module
  preprocess_image_module— same for preprocess_image.py; SKIPPED when TF
                           is absent or mocked (detected via isinstance check)
  temp_dir               — thin alias for pytest's tmp_path fixture
  valid_admin_token      — pre-signed HS256 JWT for role="admin"
  valid_user_token       — pre-signed HS256 JWT for role="user"
  mock_model             — MagicMock Keras model returning 5-class softmax
  mock_label_encoder     — MagicMock LabelEncoder with 5 known classes
  mock_pca_image         — MagicMock IncrementalPCA (256 components)
  mock_pca_text          — MagicMock IncrementalPCA (512 components)
  mock_vectorizer        — MagicMock TfidfVectorizer returning (1, 5000) array
  mock_minilm_encoder    — MagicMock SentenceTransformer returning (1, 384) float32 array
  sample_image_b64       — 32x32 RGB JPEG as base64 string (via Pillow)

Rotating log plugin (last 3 runs retained per file) :
  tests/logs/unit_tests.log        — per-test PASS/FAIL/SKIP with durations
  tests/logs/integration_tests.log — same for integration tests
  tests/logs/conftest.log          — fixture setup/teardown timeline

Design notes :
  - All service modules are loaded lazily inside fixtures to avoid TensorFlow
    import errors blocking unrelated tests at collection time.
  - sys.path is extended once at module level to expose gate-api, train-api,
    and predict-api directories to all test files.
"""
import sys
import os
import io
import base64
import importlib.util
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Python path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

for path in [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "gate-api"),
    os.path.join(PROJECT_ROOT, "train-api"),
    os.path.join(PROJECT_ROOT, "predict-api"),
]:
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Dynamic module loader for train-api services (no __init__.py)
# ---------------------------------------------------------------------------
def _import(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRAIN_SERVICES = os.path.join(PROJECT_ROOT, "train-api", "services")


# ---------------------------------------------------------------------------
# Module fixtures — lazy-loaded so tests that don't need TF can still run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def artifacts_module():
    return _import("artifacts", os.path.join(_TRAIN_SERVICES, "artifacts.py"))

@pytest.fixture(scope="session")
def pca_reducer_module():
    return _import("pca_reducer", os.path.join(_TRAIN_SERVICES, "pca_reducer.py"))

@pytest.fixture(scope="session")
def preprocess_text_module():
    return _import("preprocess_text", os.path.join(_TRAIN_SERVICES, "preprocess_text.py"))

@pytest.fixture(scope="session")
def preprocess_image_module():
    try:
        # Verify that tensorflow is real (not a MagicMock stub from another test file)
        import tensorflow as _tf
        if isinstance(_tf, MagicMock):
            pytest.skip("preprocess_image requires real TensorFlow (currently mocked)")
        return _import("preprocess_image", os.path.join(_TRAIN_SERVICES, "preprocess_image.py"))
    except ModuleNotFoundError as exc:
        pytest.skip(f"preprocess_image requires missing dependency: {exc}")

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
TEST_SECRET    = "default_secret"
TEST_ALGORITHM = "HS256"


def _make_token(username: str, role: str) -> str:
    import jwt
    payload = {
        "sub": username,
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, TEST_SECRET, algorithm=TEST_ALGORITHM)


@pytest.fixture(scope="session")
def valid_admin_token() -> str:
    return _make_token("admin", "admin")

@pytest.fixture(scope="session")
def valid_user_token() -> str:
    return _make_token("user", "user")


# ---------------------------------------------------------------------------
# Mock ML artefacts
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([[0.05, 0.60, 0.10, 0.15, 0.10]])
    return m

@pytest.fixture
def mock_label_encoder():
    le = MagicMock()
    le.classes_ = np.array([10, 40, 60, 1140, 2060])
    le.inverse_transform.return_value = np.array(["product_40"])
    le.transform.return_value = np.array([0, 1, 2])
    return le

@pytest.fixture
def mock_pca_image():
    pca = MagicMock()
    pca.n_components_ = 256   # current pca_components in params.yaml
    pca.copy = True
    pca.transform.return_value = np.zeros((1, 256), dtype=np.float32)
    return pca

@pytest.fixture
def mock_pca_text():
    pca = MagicMock()
    pca.n_components_ = 512   # current n_text_pca_components in params.yaml
    pca.copy = True
    pca.transform.return_value = np.zeros((1, 512), dtype=np.float32)
    return pca

@pytest.fixture
def mock_vectorizer():
    vec = MagicMock()
    sparse_out = MagicMock()
    sparse_out.toarray.return_value = np.zeros((1, 5000), dtype=np.float32)
    vec.transform.return_value = sparse_out
    return vec

@pytest.fixture
def mock_minilm_encoder():
    """MagicMock SentenceTransformer: encode() returns a (1, 384) float32 array."""
    enc = MagicMock()
    enc.encode.return_value = np.zeros((1, 384), dtype=np.float32)
    return enc


# ---------------------------------------------------------------------------
# Sample image — valid JPEG decodable by cv2
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_image_b64() -> str:
    """32x32 RGB JPEG, base64-encoded."""
    try:
        from PIL import Image
        img = Image.new("RGB", (32, 32), color=(180, 60, 60))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        pytest.skip("Pillow required for image fixture")


# ===========================================================================
# Rotating test-run logger
# Writes separate logs for unit tests, integration tests, and conftest events.
# Each log file retains the last 3 complete runs (separated by RUN headers).
# ===========================================================================
_MAX_RUNS = 3
_RUN_SEPARATOR = "=" * 72

_conftest_events: list[str] = []
_unit_results: list[tuple[str, str, str]] = []       # (nodeid, outcome, duration)
_integration_results: list[tuple[str, str, str]] = []


def _is_unit(nodeid: str) -> bool:
    return "unit" in nodeid.replace("\\", "/")


def _is_integration(nodeid: str) -> bool:
    return "integration" in nodeid.replace("\\", "/")


def _rotate_log(path: str, new_block: str, max_runs: int = _MAX_RUNS) -> None:
    """Append new_block to path, keeping at most max_runs blocks."""
    existing = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            existing = fh.read()

    # Split on run separators; each run block starts with the separator line
    blocks = [b for b in existing.split(_RUN_SEPARATOR) if b.strip()]
    blocks.append(new_block)
    # Keep last max_runs blocks
    blocks = blocks[-max_runs:]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(("\n" + _RUN_SEPARATOR + "\n").join(blocks))


# ---------------------------------------------------------------------------
# Conftest lifecycle hooks (used to log fixture/session events)
# ---------------------------------------------------------------------------
def pytest_configure(config):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _conftest_events.append(f"[{ts}] pytest_configure — rootdir: {config.rootpath}")


def pytest_collection_finish(session):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(session.items)
    _conftest_events.append(f"[{ts}] Collection finished — {n} test(s) collected")


@pytest.fixture(autouse=True)
def _log_fixture_setup(request):
    """Log each test's fixture setup/teardown for the conftest log."""
    ts = datetime.now().strftime("%H:%M:%S")
    nodeid = request.node.nodeid
    _conftest_events.append(f"[{ts}] SETUP   {nodeid}")
    yield
    ts2 = datetime.now().strftime("%H:%M:%S")
    _conftest_events.append(f"[{ts2}] TEARDOWN {nodeid}")


# ---------------------------------------------------------------------------
# Per-test result hook
# ---------------------------------------------------------------------------
def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    nodeid = report.nodeid
    duration = f"{report.duration:.3f}s"
    if report.passed:
        outcome = "PASS"
    elif report.failed:
        outcome = "FAIL"
    else:
        outcome = "SKIP"

    if _is_unit(nodeid):
        _unit_results.append((nodeid, outcome, duration))
    elif _is_integration(nodeid):
        _integration_results.append((nodeid, outcome, duration))


# ---------------------------------------------------------------------------
# Session finish — flush all logs
# ---------------------------------------------------------------------------
def pytest_sessionfinish(session, exitstatus):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_header = f"\nRUN: {ts}  exit={exitstatus}\n"

    # --- Unit log ---
    if _unit_results:
        lines = [run_header]
        passed = sum(1 for _, o, _ in _unit_results if o == "PASS")
        failed = sum(1 for _, o, _ in _unit_results if o == "FAIL")
        skipped = sum(1 for _, o, _ in _unit_results if o == "SKIP")
        lines.append(f"  {passed} passed  {failed} failed  {skipped} skipped\n")
        for nodeid, outcome, dur in _unit_results:
            short = nodeid.split("::")[-1]
            lines.append(f"  [{outcome}] {short:60s} ({dur})")
        _rotate_log(os.path.join(LOGS_DIR, "unit_tests.log"), "\n".join(lines))

    # --- Integration log ---
    if _integration_results:
        lines = [run_header]
        passed = sum(1 for _, o, _ in _integration_results if o == "PASS")
        failed = sum(1 for _, o, _ in _integration_results if o == "FAIL")
        skipped = sum(1 for _, o, _ in _integration_results if o == "SKIP")
        lines.append(f"  {passed} passed  {failed} failed  {skipped} skipped\n")
        for nodeid, outcome, dur in _integration_results:
            short = nodeid.split("::")[-1]
            lines.append(f"  [{outcome}] {short:60s} ({dur})")
        _rotate_log(os.path.join(LOGS_DIR, "integration_tests.log"), "\n".join(lines))

    # --- Conftest log ---
    _conftest_events.append(f"[{ts}] pytest_sessionfinish — exit={exitstatus}")
    block = "\n".join([run_header] + _conftest_events)
    _rotate_log(os.path.join(LOGS_DIR, "conftest.log"), block)
