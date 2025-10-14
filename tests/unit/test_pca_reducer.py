# tests/unit/test_pca_reducer.py
import numpy as np
import pathlib
import pytest

def test_reduce_features_saves_outputs(temp_dir, pca_reducer_module):
    """
    Test that reduce_features correctly saves PCA outputs.
    Uses the dynamically imported PCA_REDUCER module from conftest.py.
    """
    PCA_REDUCER = pca_reducer_module

    # Temporary dummy features
    text_features_path = temp_dir / "text_features.npy"
    image_features_path = temp_dir / "image_features.npy"

    np.save(text_features_path, np.random.rand(10, 20).astype(np.float32))
    np.save(image_features_path, np.random.rand(10, 30).astype(np.float32))

    save_dir = temp_dir / "output"
    save_dir.mkdir(exist_ok=True)

    # Call the reduce_features function
    X_path, pca_img_path, pca_text_path = PCA_REDUCER.reduce_features(
        text_features_path=str(text_features_path),
        image_features_path=str(image_features_path),
        n_components_img=5,
        target_dim=25,
        save_dir=str(save_dir),
        initial_batch_size=5
    )

    # Assertions: check that files exist
    assert pathlib.Path(X_path).exists()
    assert pathlib.Path(pca_img_path).exists()
    assert pathlib.Path(pca_text_path).exists()
