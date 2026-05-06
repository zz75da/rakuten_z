"""
Unit tests for train-api/services/pca_reducer.py
=================================================
Purpose : Verify that reduce_features() produces a valid combined feature
          matrix by running IncrementalPCA on synthetic text + image data,
          without touching the filesystem beyond a tmp directory.

Covered :
  TestReduceFeatures
    - Output files (X_reduced.npy, pca_image.pkl, pca_text.pkl) are created
    - X_reduced shape matches expected (n_samples, target_dim)
    - Saved PCA objects are valid sklearn IncrementalPCA instances
    - n_components_img is capped when n_samples < requested components
    - Output dtype is float32 (ready for model input)
    - Works correctly with a batch_size smaller than the dataset

Dependencies : train-api/services/pca_reducer.py, scikit-learn, numpy (no TF)
"""
import numpy as np
import pickle
import pathlib
import pytest


class TestReduceFeatures:
    def _run(self, temp_dir, pca_reducer_module, n_samples=20,
             n_text=15, n_img=12, n_img_comp=3, target_dim=10):
        text_path  = temp_dir / "text_features.npy"
        image_path = temp_dir / "image_features.npy"
        save_dir   = temp_dir / "output"
        save_dir.mkdir(exist_ok=True)

        np.save(text_path,  np.random.rand(n_samples, n_text).astype(np.float32))
        np.save(image_path, np.random.rand(n_samples, n_img).astype(np.float32))

        return pca_reducer_module.reduce_features(
            text_features_path=str(text_path),
            image_features_path=str(image_path),
            n_components_img=n_img_comp,
            target_dim=target_dim,
            save_dir=str(save_dir),
            initial_batch_size=10,
        ), save_dir

    def test_output_files_exist(self, temp_dir, pca_reducer_module):
        (x_path, img_path, txt_path), _ = self._run(temp_dir, pca_reducer_module)
        assert pathlib.Path(x_path).exists()
        assert pathlib.Path(img_path).exists()
        assert pathlib.Path(txt_path).exists()

    def test_x_reduced_shape_is_correct(self, temp_dir, pca_reducer_module):
        (x_path, _, _), _ = self._run(
            temp_dir, pca_reducer_module,
            n_samples=20, n_img_comp=3, target_dim=10
        )
        X = np.load(x_path)
        assert X.shape[0] == 20
        assert X.shape[1] == 10

    def test_pca_objects_are_valid_sklearn(self, temp_dir, pca_reducer_module):
        (_, img_path, txt_path), _ = self._run(temp_dir, pca_reducer_module)
        with open(img_path, "rb") as f:
            pca_img = pickle.load(f)
        with open(txt_path, "rb") as f:
            pca_txt = pickle.load(f)
        # Both must expose n_components_ (fitted IncrementalPCA attribute)
        assert hasattr(pca_img, "n_components_")
        assert hasattr(pca_txt, "n_components_")

    def test_n_components_img_capped_by_data(self, temp_dir, pca_reducer_module):
        """n_components_img larger than available features → capped automatically."""
        (x_path, img_path, _), _ = self._run(
            temp_dir, pca_reducer_module,
            n_samples=10, n_img=5, n_img_comp=100, target_dim=8
        )
        with open(img_path, "rb") as f:
            pca_img = pickle.load(f)
        assert pca_img.n_components_ <= 5  # capped by feature dim

    def test_output_is_float32(self, temp_dir, pca_reducer_module):
        (x_path, _, _), _ = self._run(temp_dir, pca_reducer_module)
        X = np.load(x_path)
        assert X.dtype == np.float32

    def test_small_batch_size_still_works(self, temp_dir, pca_reducer_module):
        """initial_batch_size smaller than n_components forces internal adaptation."""
        text_path  = temp_dir / "t2.npy"
        image_path = temp_dir / "i2.npy"
        save_dir   = temp_dir / "out2"
        save_dir.mkdir(exist_ok=True)
        np.save(text_path,  np.random.rand(30, 20).astype(np.float32))
        np.save(image_path, np.random.rand(30, 10).astype(np.float32))

        result = pca_reducer_module.reduce_features(
            text_features_path=str(text_path),
            image_features_path=str(image_path),
            n_components_img=3,
            target_dim=10,
            save_dir=str(save_dir),
            initial_batch_size=5,
        )
        assert pathlib.Path(result[0]).exists()
