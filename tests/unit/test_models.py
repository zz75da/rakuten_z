"""
Unit tests for ML model building and feature utilities
=======================================================
Purpose : Validate the core ML components used in the Rakuten multimodal
          classification pipeline independently of any running service.

Covered :
  TestCountVectorizer  — CountVectorizer output shape, vocab, non-negativity
  TestLabelEncoder     — encode/decode round-trip, deterministic encoding
  TestModelArchitecture (skip if TF absent)
    - Keras model builds without error (Input→Dense→Dropout→Dense/softmax)
    - Output shape matches n_classes=27
    - Softmax probabilities sum to 1 over a random batch
  TestFeatureConcatenation
    - np.hstack combines text (1024-d) and image (300-d) into 1324-d vector
    - Each modality slice is preserved after concatenation
    - Zero-padding substitution for missing modality produces correct shape

Dependencies : scikit-learn, numpy; TensorFlow optional (tests skipped if absent)
"""
import pytest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow  # noqa: F401
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

requires_tf = pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not installed")


class TestCountVectorizer:
    def test_shape_bounded_by_max_features(self):
        vec = CountVectorizer(max_features=10)
        X = vec.fit_transform(["high quality leather", "mens shoes", "leather bag"])
        assert X.shape[0] == 3
        assert X.shape[1] <= 10

    def test_known_word_in_vocabulary(self):
        vec = CountVectorizer(max_features=50)
        vec.fit(["high quality leather handbag", "running shoes"])
        assert "leather" in vec.get_feature_names_out()

    def test_transform_produces_non_negative(self):
        vec = CountVectorizer()
        X = vec.fit_transform(["apple banana", "banana cherry"])
        assert (X.toarray() >= 0).all()


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
    @requires_tf
    def test_model_builds_without_error(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(1324,))
        x = Dense(512, activation="relu")(inp)
        x = Dropout(0.5)(x)
        out = Dense(27, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        assert model is not None
        # input → dense → dropout → output
        assert len(model.layers) == 4

    @requires_tf
    def test_model_output_shape(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(1324,))
        x = Dense(512, activation="relu")(inp)
        x = Dropout(0.5)(x)
        out = Dense(27, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        assert model.output_shape == (None, 27)

    @requires_tf
    def test_model_predict_output_sums_to_one(self):
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        inp = Input(shape=(100,))
        x = Dense(64, activation="relu")(inp)
        x = Dropout(0.0)(x)     # 0 dropout for deterministic test
        out = Dense(5, activation="softmax")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        X = np.random.rand(4, 100)
        preds = model.predict(X, verbose=0)
        np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-5)


class TestFeatureConcatenation:
    def test_hstack_shape(self):
        text = np.random.rand(10, 1024)
        image = np.random.rand(10, 300)
        combined = np.hstack([text, image])
        assert combined.shape == (10, 1324)

    def test_text_slice_preserved(self):
        text = np.random.rand(5, 1024)
        image = np.random.rand(5, 300)
        combined = np.hstack([text, image])
        np.testing.assert_array_equal(combined[:, :1024], text)

    def test_image_slice_preserved(self):
        text = np.random.rand(5, 1024)
        image = np.random.rand(5, 300)
        combined = np.hstack([text, image])
        np.testing.assert_array_equal(combined[:, 1024:], image)

    def test_dummy_zeros_substitution(self):
        """When one modality is missing, zeros are used."""
        text = np.random.rand(1, 1024)
        dummy_image = np.zeros((1, 300))
        combined = np.hstack([text, dummy_image])
        assert combined.shape == (1, 1324)
        np.testing.assert_array_equal(combined[:, 1024:], 0)
