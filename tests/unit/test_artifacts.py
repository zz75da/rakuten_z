import os
import numpy as np
import pytest

def test_save_artifacts_creates_file(temp_dir, artifacts_module):
    """
    Test save_artifacts writes all expected files to a temporary folder.
    """
    ARTIFACTS = artifacts_module

    save_path = temp_dir / "artifacts"
    save_path.mkdir(exist_ok=True)

    # Dummy model
    class DummyModel:
        def save(self, path):
            with open(path, "w") as f:
                f.write("dummy")

    model = DummyModel()
    vectorizer = {"vocab": [1, 2, 3]}
    pca_models = {"image": np.eye(3), "text": np.eye(2)}
    label_encoder = {"classes": [0, 1]}

    # --- Monkeypatch ARTIFACTS_PATH ---
    original_path = ARTIFACTS.ARTIFACTS_PATH
    ARTIFACTS.ARTIFACTS_PATH = str(save_path)

    try:
        ARTIFACTS.save_artifacts(
            model=model,
            vectorizer=vectorizer,
            pca_models=pca_models,
            label_encoder=label_encoder
        )
    finally:
        # Restore original path
        ARTIFACTS.ARTIFACTS_PATH = original_path

    # --- Check that all expected files exist ---
    for fname in [
        "neural_network_model.h5",
        "text_vectorizer.pkl",
        "pca_image.pkl",
        "pca_text.pkl",
        "label_encoder.pkl"
    ]:
        assert os.path.exists(save_path / fname)
