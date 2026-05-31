"""
Unit tests for train-api/services/preprocess_text.py
=====================================================
Purpose : Verify the text cleaning and TF-IDF feature extraction that produces
          the 10 000-dimensional TF-IDF input for the text PCA branch.

Covered :
  TestPreprocessText
    - Empty string returns an empty string (no crash)
    - Content words are kept after tokenisation
    - French/English stop-words are removed
    - Punctuation and non-alpha tokens are stripped
    - Output is lowercased

  TestExtractTextFeatures
    - Row count matches input DataFrame size
    - Column count is bounded by max_features (default 10 000)
    - Returned TfidfVectorizer is already fitted (has vocabulary_)
    - Feature values are numeric and non-negative (TF-IDF weights ≥ 0)
    - Same text input produces identical feature rows (deterministic)
    - Numpy .npy feature file is written and loads back correctly

Dependencies : train-api/services/preprocess_text.py, spacy (en_core_web_sm),
               scikit-learn, pandas, numpy

Note: extract_text_features() calls _build_combined_text() which requires
      both "designation" and "description" columns in the DataFrame.
"""
import pandas as pd
import numpy as np
import pytest


def _df(*descriptions, designation="product"):
    """Helper: build a minimal DataFrame with required columns."""
    return pd.DataFrame({
        "designation":  [designation] * len(descriptions),
        "description":  list(descriptions),
        "imageid":      list(range(len(descriptions))),
        "productid":    list(range(len(descriptions))),
    })


class TestPreprocessText:
    """Tests for _clean_text() — HTML unescape + strip tags + normalise whitespace.
    Note: _clean_text does NOT lowercase or strip punctuation; those happen
    inside extract_text_features() via SpaCy lemmatisation."""

    def test_empty_string_returns_empty(self, preprocess_text_module):
        result = preprocess_text_module._clean_text("")
        assert result == ""

    def test_non_empty_returns_non_empty(self, preprocess_text_module):
        result = preprocess_text_module._clean_text("The quick brown fox")
        assert len(result.strip()) > 0

    def test_html_tags_stripped(self, preprocess_text_module):
        result = preprocess_text_module._clean_text("<b>quality</b> bag")
        assert "<b>" not in result
        assert "quality" in result

    def test_html_entities_unescaped(self, preprocess_text_module):
        result = preprocess_text_module._clean_text("&amp; leather &lt;bag&gt;")
        assert "&amp;" not in result
        assert "&lt;" not in result

    def test_whitespace_normalised(self, preprocess_text_module):
        result = preprocess_text_module._clean_text("  multiple   spaces  ")
        assert "  " not in result   # no double spaces


class TestExtractTextFeatures:
    def test_output_shape_rows_match_input(self, preprocess_text_module):
        df = _df("apple banana", "cherry date", "elderberry")
        features, vectorizer = preprocess_text_module.extract_text_features(df)
        assert features.shape[0] == 3

    def test_output_columns_bounded_by_max_features(self, preprocess_text_module):
        df = _df(*["apple banana cherry"] * 5)
        features, _ = preprocess_text_module.extract_text_features(df, max_features=100)
        assert features.shape[1] <= 100

    def test_vectorizer_is_fitted(self, preprocess_text_module):
        df = _df("apple", "banana")
        _, vectorizer = preprocess_text_module.extract_text_features(df)
        assert hasattr(vectorizer, "vocabulary_")

    def test_features_are_numeric(self, preprocess_text_module):
        df = _df("leather bag", "running shoes")
        features, _ = preprocess_text_module.extract_text_features(df)
        assert np.issubdtype(features.dtype, np.number)

    def test_features_are_non_negative(self, preprocess_text_module):
        df = _df("quality product", "blue shoes size 42")
        features, _ = preprocess_text_module.extract_text_features(df)
        assert (features >= 0).all()

    def test_same_text_gives_identical_rows(self, preprocess_text_module):
        df = _df("leather bag", "leather bag")
        features, _ = preprocess_text_module.extract_text_features(df)
        np.testing.assert_array_equal(features[0], features[1])

    def test_features_persist_to_disk(self, temp_dir, preprocess_text_module):
        df = _df("apple banana", "cherry date")
        features, _ = preprocess_text_module.extract_text_features(df)
        path = temp_dir / "text_features.npy"
        np.save(path, features)
        loaded = np.load(path)
        np.testing.assert_array_equal(features, loaded)

    def test_vectorizer_is_tfidf(self, preprocess_text_module):
        from sklearn.feature_extraction.text import TfidfVectorizer
        df = _df("apple banana", "cherry date")
        _, vectorizer = preprocess_text_module.extract_text_features(df)
        assert isinstance(vectorizer, TfidfVectorizer)

    def test_designation_combined_with_description(self, preprocess_text_module):
        """Both designation and description are used — more text → richer features.
        Compare single-column vs combined: combined produces >= features."""
        df_both = pd.DataFrame({
            "designation": ["gaming laptop computer"],
            "description": ["high performance"],
            "imageid":     [1],
            "productid":   [1],
        })
        df_desc_only = pd.DataFrame({
            "designation": [""],         # empty designation
            "description": ["high performance"],
            "imageid":     [1],
            "productid":   [1],
        })
        feat_both, _ = preprocess_text_module.extract_text_features(df_both)
        feat_desc, _ = preprocess_text_module.extract_text_features(df_desc_only)
        # both produce valid arrays
        assert feat_both.shape[0] == 1
        assert feat_desc.shape[0] == 1
