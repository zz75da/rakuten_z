# tests/unit/test_preprocess_text.py
import pandas as pd
import numpy as np
import pytest

def test_preprocess_text_empty_string(preprocess_text_module):
    """
    Test that preprocessing an empty string returns an empty string.
    """
    PREPROCESS_TEXT = preprocess_text_module
    assert PREPROCESS_TEXT.preprocess_text("") == ""

def test_preprocess_text_basic(preprocess_text_module):
    """
    Test that basic text preprocessing works as expected.
    """
    PREPROCESS_TEXT = preprocess_text_module
    text = "The quick brown fox jumps!"
    result = PREPROCESS_TEXT.preprocess_text(text)
    assert "quick" in result
    assert "fox" in result

def test_extract_text_features_shape(temp_dir, preprocess_text_module):
    """
    Test extraction of text features from a mini dataframe.
    """
    PREPROCESS_TEXT = preprocess_text_module
    df = pd.DataFrame({"description": ["apple banana", "cherry date"]})
    features, vectorizer = PREPROCESS_TEXT.extract_text_features(df)

    assert features.shape[0] == df.shape[0]
    assert features.shape[1] <= 5000

    # Save features temporarily
    np.save(temp_dir / "text_features.npy", features)
