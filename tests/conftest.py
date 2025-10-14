# tests/conftest.py

import sys
import os
import importlib.util
import pytest

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# --- Helper to dynamically import modules by path ---
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- TRAIN-API modules ---
TRAIN_API_SERVICES = os.path.join(PROJECT_ROOT, "train-api", "services")

ARTIFACTS = import_module_from_path(
    "artifacts",
    os.path.join(TRAIN_API_SERVICES, "artifacts.py")
)

PREPROCESS_TEXT = import_module_from_path(
    "preprocess_text",
    os.path.join(TRAIN_API_SERVICES, "preprocess_text.py")
)

PREPROCESS_IMAGE = import_module_from_path(
    "preprocess_image",
    os.path.join(TRAIN_API_SERVICES, "preprocess_image.py")
)

PCA_REDUCER = import_module_from_path(
    "pca_reducer",
    os.path.join(TRAIN_API_SERVICES, "pca_reducer.py")
)

# --- GATE-API modules (optional) ---
GATE_API_SERVICES = os.path.join(PROJECT_ROOT, "gate-api", "services")
GATE_MODULES = {}
if os.path.exists(GATE_API_SERVICES):
    for file in os.listdir(GATE_API_SERVICES):
        if file.endswith(".py") and not file.startswith("__"):
            name = file[:-3]
            GATE_MODULES[name] = import_module_from_path(
                name, os.path.join(GATE_API_SERVICES, file)
            )

# --- PREDICT-API modules (optional) ---
PREDICT_API_SERVICES = os.path.join(PROJECT_ROOT, "predict-api", "services")
PREDICT_MODULES = {}
if os.path.exists(PREDICT_API_SERVICES):
    for file in os.listdir(PREDICT_API_SERVICES):
        if file.endswith(".py") and not file.startswith("__"):
            name = file[:-3]
            PREDICT_MODULES[name] = import_module_from_path(
                name, os.path.join(PREDICT_API_SERVICES, file)
            )

# --- Pytest fixtures for temp path ---
@pytest.fixture
def temp_dir(tmp_path):
    """Fixture that provides a temporary directory for tests."""
    return tmp_path

# --- Pytest fixtures for imported modules ---
@pytest.fixture
def artifacts_module():
    return ARTIFACTS

@pytest.fixture
def pca_reducer_module():
    return PCA_REDUCER

@pytest.fixture
def preprocess_text_module():
    return PREPROCESS_TEXT

@pytest.fixture
def preprocess_image_module():
    return PREPROCESS_IMAGE

@pytest.fixture
def gate_modules():
    return GATE_MODULES

@pytest.fixture
def predict_modules():
    return PREDICT_MODULES
