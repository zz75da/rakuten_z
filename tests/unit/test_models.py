"""
Unit tests for ML model building and feature utilities
=======================================================
Purpose : Validate the core ML components used in the Rakuten multimodal
          classification pipeline independently of any running service.

Current architecture (params.yaml):
  image PCA : 256-d   (pca_components)
  text PCA  : 512-d   (n_text_pca_components, CV encoder only)
  Late fusion input dims:
    CV    : 512 (text PCA) + 256 (image PCA) = 768
    CLIP  : 512 (CLIP text) + 256 (image PCA) = 768
    MiniLM: 384 (sentence-transformers) + 256 (image PCA) = 640
    mpnet : 768 (sentence-transformers) + 256 (image PCA) = 1024

Covered :
  TestTfidfVectorizer    — TfidfVectorizer output shape, vocab, sublinear scaling
  TestLabelEncoder       — encode/decode round-trip, deterministic encoding
  TestModelArchitecture  — late-fusion Keras model builds, output shape, softmax sums
  TestFeatureConcatenation — per-encoder hstack dimensions

Dependencies : scikit-learn, numpy; TensorFlow optional (tests skipped if absent)
"""
import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow  # noqa: F401
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

requires_tf = pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not installed")

# ── Current input dimensions ─────────────────────────────────────────────────
_IMAGE_DIM  = 256   # pca_components
_TEXT_PCA   = 512   # n_text_pca_components (CV only)
_MINILM_DIM = 384
_MPNET_DIM  = 768
_CLIP_DIM   = 512

_CV_INPUT    = _TEXT_PCA + _IMAGE_DIM    # 768
_CLIP_INPUT  = _CLIP_DIM  + _IMAGE_DIM   # 768
_MINILM_INPUT= _MINILM_DIM + _IMAGE_DIM  # 640
_MPNET_INPUT = _MPNET_DIM  + _IMAGE_DIM  # 1024


class TestTfidfVectorizer:
    def test_shape_bounded_by_max_features(self):
        vec = TfidfVectorizer(max_features=10)
        X = vec.fit_transform(["high quality leather", "mens shoes", "leather bag"])
        assert X.shape[0] == 3
        assert X.shape[1] <= 10

    def test_known_word_in_vocabulary(self):
        vec = TfidfVectorizer(max_features=50)
        vec.fit(["high quality leather handbag", "running shoes"])
        assert "leather" in vec.get_feature_names_out()

    def test_transform_produces_non_negative(self):
        vec = TfidfVectorizer(sublinear_tf=True)
        X = vec.fit_transform(["apple banana", "banana cherry"])
        assert (X.toarray() >= 0).all()

    def test_sublinear_tf_reduces_high_frequency_weight(self):
        """sublinear_tf=True (log(1+tf)) should reduce weight of very frequent terms."""
        vec_linear = TfidfVectorizer(sublinear_tf=False)
        vec_log    = TfidfVectorizer(sublinear_tf=True)
        docs = ["apple " * 10 + "banana", "cherry date"]
        X_linear = vec_linear.fit_transform(docs).toarray()
        X_log    = vec_log.fit_transform(docs).toarray()
        # "apple" column max should be lower with sublinear_tf
        apple_idx = list(vec_log.vocabulary_.keys()).index("apple") \
                    if "apple" in vec_log.vocabulary_ else None
        if apple_idx is not None:
            assert X_log[0, apple_idx] <= X_linear[0, apple_idx]


class TestLabelEncoder:
    def test_encode_decode_roundtrip(self):
        enc = LabelEncoder()
        labels = [40, 60, 1140, 40, 60]
        encoded = enc.fit_transform(labels)
        assert len(enc.classes_) == 3
        decoded = enc.inverse_transform(encoded)
        np.testing.assert_array_equal(decoded, labels)

    def test_same_class_same_encoding(self):
        enc = LabelEncoder()
        encoded = enc.fit_transform([40, 60, 1140, 40, 60])
        assert encoded[0] == encoded[3]  # both class 40
        assert encoded[1] == encoded[4]  # both class 60


class TestModelArchitecture:
    """Late-fusion dense head — CV encoder (input_dim=768)."""

    @requires_tf
    def test_model_builds_without_error(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(_CV_INPUT,))
        x = Dense(512, activation="relu")(inp)
        x = Dropout(0.35)(x)
        out = Dense(27, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        assert model.input_shape == (None, _CV_INPUT)
        assert model.output_shape == (None, 27)

    @requires_tf
    def test_model_predict_output_sums_to_one(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(100,))
        x = Dense(64, activation="relu")(inp)
        x = Dropout(0.0)(x)
        out = Dense(5, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        preds = model.predict(np.random.rand(4, 100), verbose=0)
        np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-5)


class TestFeatureConcatenation:
    """CV encoder: text_pca=512, image_pca=256 → combined=768."""

    def test_cv_hstack_shape(self):
        text  = np.random.rand(10, _TEXT_PCA)
        image = np.random.rand(10, _IMAGE_DIM)
        assert np.hstack([text, image]).shape == (10, _CV_INPUT)

    def test_text_slice_preserved(self):
        text  = np.random.rand(5, _TEXT_PCA)
        image = np.random.rand(5, _IMAGE_DIM)
        combined = np.hstack([text, image])
        np.testing.assert_array_equal(combined[:, :_TEXT_PCA], text)

    def test_image_slice_preserved(self):
        text  = np.random.rand(5, _TEXT_PCA)
        image = np.random.rand(5, _IMAGE_DIM)
        combined = np.hstack([text, image])
        np.testing.assert_array_equal(combined[:, _TEXT_PCA:], image)

    def test_dummy_zeros_substitution(self):
        text        = np.random.rand(1, _TEXT_PCA)
        dummy_image = np.zeros((1, _IMAGE_DIM))
        combined    = np.hstack([text, dummy_image])
        assert combined.shape == (1, _CV_INPUT)
        np.testing.assert_array_equal(combined[:, _TEXT_PCA:], 0)


class TestFeatureConcatenationMiniLM:
    """MiniLM encoder: text=384, image_pca=256 → combined=640."""

    def test_hstack_shape(self):
        assert np.hstack([np.random.rand(10, _MINILM_DIM),
                          np.random.rand(10, _IMAGE_DIM)]).shape == (10, _MINILM_INPUT)

    def test_text_slice_preserved(self):
        text  = np.random.rand(5, _MINILM_DIM)
        image = np.random.rand(5, _IMAGE_DIM)
        combined = np.hstack([text, image])
        np.testing.assert_array_equal(combined[:, :_MINILM_DIM], text)

    def test_dummy_zeros_substitution(self):
        combined = np.hstack([np.zeros((1, _MINILM_DIM)),
                               np.random.rand(1, _IMAGE_DIM)])
        assert combined.shape == (1, _MINILM_INPUT)
        np.testing.assert_array_equal(combined[:, :_MINILM_DIM], 0)


class TestFeatureConcatenationClip:
    """CLIP encoder: text=512, image_pca=256 → combined=768."""

    def test_hstack_shape(self):
        assert np.hstack([np.random.rand(10, _CLIP_DIM),
                          np.random.rand(10, _IMAGE_DIM)]).shape == (10, _CLIP_INPUT)

    def test_text_slice_preserved(self):
        text  = np.random.rand(5, _CLIP_DIM)
        image = np.random.rand(5, _IMAGE_DIM)
        combined = np.hstack([text, image])
        np.testing.assert_array_equal(combined[:, :_CLIP_DIM], text)


class TestFeatureConcatenationMpnet:
    """mpnet encoder: text=768, image_pca=256 → combined=1024."""

    def test_hstack_shape(self):
        assert np.hstack([np.random.rand(10, _MPNET_DIM),
                          np.random.rand(10, _IMAGE_DIM)]).shape == (10, _MPNET_INPUT)

    def test_text_slice_preserved(self):
        text  = np.random.rand(5, _MPNET_DIM)
        image = np.random.rand(5, _IMAGE_DIM)
        combined = np.hstack([text, image])
        np.testing.assert_array_equal(combined[:, :_MPNET_DIM], text)

    def test_dummy_zeros_substitution(self):
        combined = np.hstack([np.zeros((1, _MPNET_DIM)),
                               np.random.rand(1, _IMAGE_DIM)])
        assert combined.shape == (1, _MPNET_INPUT)


class TestModelArchitectureMiniLM:
    """MiniLM: input_dim=640 (384 + 256)."""

    @requires_tf
    def test_model_builds_with_minilm_input(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(_MINILM_INPUT,))
        x = Dense(512, activation="relu")(inp)
        x = Dropout(0.35)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.25)(x)
        out = Dense(27, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        assert model.input_shape  == (None, _MINILM_INPUT)
        assert model.output_shape == (None, 27)

    @requires_tf
    def test_minilm_model_output_sums_to_one(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(_MINILM_INPUT,))
        x   = Dense(64, activation="relu")(inp)
        x   = Dropout(0.0)(x)
        out = Dense(27, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        preds = model.predict(np.random.rand(4, _MINILM_INPUT), verbose=0)
        np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-5)


class TestModelArchitectureClip:
    """CLIP: input_dim=768 (512 + 256)."""

    @requires_tf
    def test_model_builds_with_clip_input(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(_CLIP_INPUT,))
        x = Dense(512, activation="relu")(inp)
        x = Dropout(0.35)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.25)(x)
        out = Dense(27, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        assert model.input_shape  == (None, _CLIP_INPUT)
        assert model.output_shape == (None, 27)

    @requires_tf
    def test_clip_model_output_sums_to_one(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(_CLIP_INPUT,))
        x   = Dense(64, activation="relu")(inp)
        x   = Dropout(0.0)(x)
        out = Dense(27, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        preds = model.predict(np.random.rand(4, _CLIP_INPUT), verbose=0)
        np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-5)


class TestModelArchitectureMpnet:
    """mpnet: input_dim=1024 (768 + 256)."""

    @requires_tf
    def test_model_builds_with_mpnet_input(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(_MPNET_INPUT,))
        x = Dense(768, activation="relu")(inp)
        x = Dropout(0.4)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.3)(x)
        out = Dense(27, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        assert model.input_shape  == (None, _MPNET_INPUT)
        assert model.output_shape == (None, 27)

    @requires_tf
    def test_mpnet_model_output_sums_to_one(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(_MPNET_INPUT,))
        x   = Dense(64, activation="relu")(inp)
        x   = Dropout(0.0)(x)
        out = Dense(27, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        preds = model.predict(np.random.rand(4, _MPNET_INPUT), verbose=0)
        np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-5)
