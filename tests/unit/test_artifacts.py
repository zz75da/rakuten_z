"""
Unit tests for train-api/services/artifacts.py
================================================
Purpose : Verify that save_artifacts() correctly persists all ML artefacts
          (Keras model, vectorizer, PCA objects, label encoder) to disk.

Covered :
  TestSaveArtifacts
    - All expected output files are created (.keras + .pkl set)
    - skip_existing=True preserves an existing file unchanged
    - skip_existing=False (force-overwrite) replaces a stale file
    - Pickle files survive a write-read round-trip without corruption

Dependencies : train-api/services/artifacts.py (no network, no TF runtime)
"""
import os
import pickle
import pytest
import numpy as np


class _DummyModel:
    def save(self, path):
        with open(path, "w") as f:
            f.write("dummy-keras-model")


def _make_kwargs(skip_existing=False):
    return dict(
        model=_DummyModel(),
        vectorizer={"vocab": [1, 2, 3]},
        pca_models={"image": np.eye(3), "text": np.eye(2)},
        label_encoder={"classes": [0, 1]},
        skip_existing=skip_existing,
    )


class TestSaveArtifacts:
    def test_creates_all_expected_files(self, temp_dir, artifacts_module):
        save_path = temp_dir / "arts"
        save_path.mkdir()
        orig = artifacts_module.ARTIFACTS_PATH
        artifacts_module.ARTIFACTS_PATH = str(save_path)
        try:
            artifacts_module.save_artifacts(**_make_kwargs())
        finally:
            artifacts_module.ARTIFACTS_PATH = orig

        # Format is now .keras, not .h5
        for fname in ["neural_network_model.keras", "text_vectorizer.pkl",
                      "pca_image.pkl", "pca_text.pkl", "label_encoder.pkl"]:
            assert (save_path / fname).exists(), f"Missing: {fname}"

    def test_skip_existing_preserves_original(self, temp_dir, artifacts_module):
        save_path = temp_dir / "arts_skip"
        save_path.mkdir()
        sentinel = save_path / "pca_image.pkl"
        with open(sentinel, "wb") as f:
            pickle.dump("ORIGINAL", f)

        orig = artifacts_module.ARTIFACTS_PATH
        artifacts_module.ARTIFACTS_PATH = str(save_path)
        try:
            artifacts_module.save_artifacts(**_make_kwargs(skip_existing=True))
        finally:
            artifacts_module.ARTIFACTS_PATH = orig

        with open(sentinel, "rb") as f:
            assert pickle.load(f) == "ORIGINAL"

    def test_force_overwrite_replaces_stale_file(self, temp_dir, artifacts_module):
        save_path = temp_dir / "arts_force"
        save_path.mkdir()
        sentinel = save_path / "pca_image.pkl"
        with open(sentinel, "wb") as f:
            pickle.dump("STALE", f)

        orig = artifacts_module.ARTIFACTS_PATH
        artifacts_module.ARTIFACTS_PATH = str(save_path)
        try:
            artifacts_module.save_artifacts(**_make_kwargs(skip_existing=False))
        finally:
            artifacts_module.ARTIFACTS_PATH = orig

        with open(sentinel, "rb") as f:
            val = pickle.load(f)
        # After overwrite the value is a numpy array (np.eye(3)), not the string "STALE"
        assert not (isinstance(val, str) and val == "STALE")

    def test_pkl_files_roundtrip_cleanly(self, temp_dir, artifacts_module):
        save_path = temp_dir / "arts_pkl"
        save_path.mkdir()
        orig = artifacts_module.ARTIFACTS_PATH
        artifacts_module.ARTIFACTS_PATH = str(save_path)
        try:
            artifacts_module.save_artifacts(**_make_kwargs())
        finally:
            artifacts_module.ARTIFACTS_PATH = orig

        for fname in ("text_vectorizer.pkl", "pca_image.pkl",
                      "pca_text.pkl", "label_encoder.pkl"):
            with open(save_path / fname, "rb") as f:
                assert pickle.load(f) is not None
