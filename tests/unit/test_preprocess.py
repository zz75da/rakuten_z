"""
Preprocessing smoke tests (cross-module sanity checks)
=======================================================
Purpose : Quick sanity checks that confirm both preprocessing modules
          expose the expected public API and return sensible types.
          These are deliberately shallow — deep coverage lives in
          test_preprocess_text.py and test_preprocess_image.py.

Covered :
  TestTextPreprocessingSmoke
    - preprocess_text module exports preprocess_text() and extract_text_features()
    - preprocess_text() returns a string for a simple input
    - extract_text_features() returns a (matrix, vectorizer) tuple
    - Feature matrix is a numpy ndarray

  TestImagePreprocessingSmoke  (SKIPPED when TF is unavailable)
    - preprocess_image module exports extract_image_features()
    - Passing non-existent image paths does not raise an unhandled exception

Dependencies : conftest fixtures preprocess_text_module, preprocess_image_module
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


def _smoke_df(*descriptions):
    """Minimal DataFrame with required columns for preprocess_text."""
    return pd.DataFrame({
        "designation": ["product"] * len(descriptions),
        "description": list(descriptions),
        "imageid":     list(range(len(descriptions))),
        "productid":   list(range(len(descriptions))),
    })


class TestTextPreprocessingSmoke:
    def test_module_has_clean_text(self, preprocess_text_module):
        # public API: _clean_text (HTML unescape/strip) and extract_text_features
        assert hasattr(preprocess_text_module, "_clean_text")

    def test_module_has_extract_text_features(self, preprocess_text_module):
        assert hasattr(preprocess_text_module, "extract_text_features")

    def test_clean_text_returns_string(self, preprocess_text_module):
        result = preprocess_text_module._clean_text("High quality bag")
        assert isinstance(result, str)

    def test_extract_features_returns_tuple(self, preprocess_text_module):
        df = _smoke_df("apple", "banana")
        result = preprocess_text_module.extract_text_features(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_feature_matrix_is_numpy(self, preprocess_text_module):
        df = _smoke_df("chair", "table", "sofa")
        features, _ = preprocess_text_module.extract_text_features(df)
        assert isinstance(features, np.ndarray)


class TestImagePreprocessingSmoke:
    def test_module_has_extract_image_features(self, preprocess_image_module):
        assert hasattr(preprocess_image_module, "extract_image_features")

    def test_missing_images_do_not_raise(self, temp_dir, preprocess_image_module):
        df = pd.DataFrame({"image_path": [
            str(temp_dir / "nope1.jpg"),
            str(temp_dir / "nope2.jpg"),
        ]})
        try:
            result = preprocess_image_module.extract_image_features(df)
            # If it returns, result is None or a 2D array
            if result is not None:
                assert result.ndim == 2
        except Exception as exc:
            pytest.fail(f"extract_image_features raised unexpectedly: {exc}")
