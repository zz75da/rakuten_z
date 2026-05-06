"""
Unit tests for train-api/services/preprocess_text.py
=====================================================
Purpose : Verify the text cleaning and CountVectorizer feature extraction
          that produces the 5 000-dimensional BoW input for the text PCA branch.

Covered :
  TestPreprocessText
    - Empty string returns an empty string (no crash)
    - Content words are kept after tokenisation
    - French/English stop-words are removed
    - Punctuation and non-alpha tokens are stripped
    - Output is lowercased

  TestExtractTextFeatures
    - Row count matches input DataFrame size
    - Column count is bounded by max_features (default 5 000)
    - Returned CountVectorizer is already fitted (has vocabulary_)
    - Feature values are numeric and non-negative (raw counts)
    - Same text input produces identical feature rows (deterministic)
    - Numpy .npy feature file is written and loads back correctly

Dependencies : train-api/services/preprocess_text.py, spacy (en_core_web_sm),
               scikit-learn, pandas, numpy
"""
import pandas as pd
import numpy as np
import pytest


class TestPreprocessText:
    def test_empty_string_returns_empty(self, preprocess_text_module):
        result = preprocess_text_module.preprocess_text("")
        assert result == ""

    def test_content_words_kept(self, preprocess_text_module):
        result = preprocess_text_module.preprocess_text("The quick brown fox jumps!")
        # stopwords removed, content words lemmatised and present
        assert "quick" in result or "fox" in result or "jump" in result

    def test_stopwords_removed(self, preprocess_text_module):
        result = preprocess_text_module.preprocess_text("the and or is are")
        # common English stopwords should be stripped
        assert result.strip() == ""

    def test_punctuation_stripped(self, preprocess_text_module):
        result = preprocess_text_module.preprocess_text("hello!!! world???")
        assert "!" not in result
        assert "?" not in result

    def test_output_is_lowercase(self, preprocess_text_module):
        result = preprocess_text_module.preprocess_text("HIGH QUALITY Leather BAG")
        assert result == result.lower()

    def test_non_alpha_tokens_removed(self, preprocess_text_module):
        result = preprocess_text_module.preprocess_text("product123 abc 456")
        assert "123" not in result
        assert "456" not in result


class TestExtractTextFeatures:
    def test_output_shape_rows_match_input(self, preprocess_text_module):
        df = pd.DataFrame({"description": ["apple banana", "cherry date", "elderberry"]})
        features, vectorizer = preprocess_text_module.extract_text_features(df)
        assert features.shape[0] == 3

    def test_output_columns_bounded_by_max_features(self, preprocess_text_module):
        df = pd.DataFrame({"description": ["apple banana cherry"] * 5})
        features, _ = preprocess_text_module.extract_text_features(df)
        assert features.shape[1] <= 5000

    def test_vectorizer_is_fitted(self, preprocess_text_module):
        df = pd.DataFrame({"description": ["apple", "banana"]})
        _, vectorizer = preprocess_text_module.extract_text_features(df)
        # A fitted vectorizer exposes vocabulary_
        assert hasattr(vectorizer, "vocabulary_")

    def test_features_are_numeric(self, preprocess_text_module):
        df = pd.DataFrame({"description": ["leather bag", "running shoes"]})
        features, _ = preprocess_text_module.extract_text_features(df)
        assert np.issubdtype(features.dtype, np.number)

    def test_features_are_non_negative(self, preprocess_text_module):
        df = pd.DataFrame({"description": ["quality product", "blue shoes size 42"]})
        features, _ = preprocess_text_module.extract_text_features(df)
        assert (features >= 0).all()

    def test_same_text_gives_identical_rows(self, preprocess_text_module):
        df = pd.DataFrame({"description": ["leather bag", "leather bag"]})
        features, _ = preprocess_text_module.extract_text_features(df)
        np.testing.assert_array_equal(features[0], features[1])

    def test_features_persist_to_disk(self, temp_dir, preprocess_text_module):
        df = pd.DataFrame({"description": ["apple banana", "cherry date"]})
        features, _ = preprocess_text_module.extract_text_features(df)
        path = temp_dir / "text_features.npy"
        np.save(path, features)
        loaded = np.load(path)
        np.testing.assert_array_equal(features, loaded)
