# tests/integration/test_mini_pipeline.py
import pytest
import numpy as np
import pandas as pd
import os

@pytest.mark.integration
def test_mini_pipeline(tmp_path, preprocess_text_module, pca_reducer_module):
    """
    End-to-end mini pipeline test: extract text features + reduce features.
    """
    PREPROCESS_TEXT = preprocess_text_module
    PCA_REDUCER = pca_reducer_module

    # --- Mini text dataset ---
    df = pd.DataFrame({"description": ["apple", "banana", "cherry"]})
    text_features, vectorizer = PREPROCESS_TEXT.extract_text_features(df)
    np.save(tmp_path / "text.npy", text_features)

    # --- Mini image dataset ---
    image_features = np.random.rand(3, 2048).astype(np.float32)
    np.save(tmp_path / "image.npy", image_features)

    # --- Run PCA reducer ---
    X_path, pca_img_path, pca_text_path = PCA_REDUCER.reduce_features(
        text_features_path=str(tmp_path / "text.npy"),
        image_features_path=str(tmp_path / "image.npy"),
        n_components_img=2,
        target_dim=10,
        save_dir=str(tmp_path),
        initial_batch_size=2
    )

    # --- Assertions ---
    X = np.load(X_path)
    assert X.shape == (3, 10)
    assert os.path.exists(pca_img_path)
    assert os.path.exists(pca_text_path)
