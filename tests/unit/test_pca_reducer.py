"""
Unit tests for train-api/services/pca_reducer.py
=================================================
Purpose : Verify that reduce_features() produces a valid combined feature
          matrix by running IncrementalPCA on synthetic text + image data,
          without touching the filesystem beyond a tmp directory.

Covered :
  TestReduceFeatures
    - Output files (X_reduced_{enc}_{n}.npy, pca_image_{n}.pkl, pca_text_{n}.pkl) are created
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

    def test_encoder_specific_filename(self, temp_dir, pca_reducer_module):
        """X_reduced is named X_reduced_{encoder}_{n_components_img}.npy (versioned)."""
        (x_path, img_path, _), save_dir = self._run(
            temp_dir, pca_reducer_module, n_img_comp=3)
        name = pathlib.Path(x_path).name
        # versioned format: X_reduced_countvectorizer_3.npy
        assert name == "X_reduced_countvectorizer_3.npy", f"Unexpected filename: {name}"
        assert pathlib.Path(img_path).name == "pca_image_3.pkl"
        assert not (save_dir / "X_reduced.npy").exists(), \
            "Legacy unversioned X_reduced.npy must not be written"

    def test_mpnet_pass_through_skips_text_pca(self, temp_dir, pca_reducer_module):
        """mpnet: 768-d text features pass through unchanged, pca_text_path is None."""
        text_path  = temp_dir / "text_mpnet.npy"
        image_path = temp_dir / "image_mpnet.npy"
        save_dir   = temp_dir / "out_mpnet"
        save_dir.mkdir(exist_ok=True)
        n, text_dim, img_dim = 20, 768, 12

        np.save(text_path,  np.random.rand(n, text_dim).astype(np.float32))
        np.save(image_path, np.random.rand(n, img_dim).astype(np.float32))

        x_path, img_path, txt_path = pca_reducer_module.reduce_features(
            text_features_path=str(text_path),
            image_features_path=str(image_path),
            n_components_img=3,
            save_dir=str(save_dir),
            initial_batch_size=10,
            text_encoder="mpnet",
        )

        assert txt_path is None, "pca_text_path must be None for mpnet"
        X = np.load(x_path)
        assert X.shape == (n, text_dim + 3)   # 768 + 3 = 771
        assert X.dtype == np.float32
        # versioned filename
        assert pathlib.Path(x_path).name == "X_reduced_mpnet_3.npy"

    def test_minilm_pass_through_skips_text_pca(self, temp_dir, pca_reducer_module):
        """MiniLM: text features pass through unchanged, pca_text_path is None."""
        text_path  = temp_dir / "text_minilm.npy"
        image_path = temp_dir / "image.npy"
        save_dir   = temp_dir / "out_minilm"
        save_dir.mkdir(exist_ok=True)
        n, text_dim, img_dim = 20, 384, 12

        np.save(text_path,  np.random.rand(n, text_dim).astype(np.float32))
        np.save(image_path, np.random.rand(n, img_dim).astype(np.float32))

        x_path, img_path, txt_path = pca_reducer_module.reduce_features(
            text_features_path=str(text_path),
            image_features_path=str(image_path),
            n_components_img=3,
            save_dir=str(save_dir),
            initial_batch_size=10,
            text_encoder="minilm",
        )

        assert txt_path is None, "pca_text_path must be None for MiniLM"
        X = np.load(x_path)
        # MiniLM text (384) + image PCA (3) = 387
        assert X.shape == (n, text_dim + 3)
        assert X.dtype == np.float32

    def test_clip_pass_through_skips_text_pca(self, temp_dir, pca_reducer_module):
        """CLIP: 512-d text features pass through unchanged, pca_text_path is None."""
        text_path  = temp_dir / "text_clip.npy"
        image_path = temp_dir / "image_clip.npy"
        save_dir   = temp_dir / "out_clip"
        save_dir.mkdir(exist_ok=True)
        n, text_dim, img_dim = 20, 512, 12

        np.save(text_path,  np.random.rand(n, text_dim).astype(np.float32))
        np.save(image_path, np.random.rand(n, img_dim).astype(np.float32))

        x_path, img_path, txt_path = pca_reducer_module.reduce_features(
            text_features_path=str(text_path),
            image_features_path=str(image_path),
            n_components_img=3,
            save_dir=str(save_dir),
            initial_batch_size=10,
            text_encoder="clip",
        )

        assert txt_path is None, "pca_text_path must be None for CLIP"
        X = np.load(x_path)
        # CLIP text (512) + image PCA (3) = 515
        assert X.shape == (n, text_dim + 3)
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
