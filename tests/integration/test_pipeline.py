"""
Integration tests: text preprocessing → PCA reduction → feature matrix
=======================================================================
Purpose : Run the full feature-engineering sub-pipeline end-to-end using
          real scikit-learn / numpy code on synthetic in-memory data.
          No network calls, no Docker, no GPU required.

Covered :
  TestTextToPCAMiniPipeline
    - full pipeline : extract_text_features() → reduce_features()
      produces a valid X_reduced.npy and two serialised PCA objects
    - PCA objects can transform unseen samples at inference time
      (simulates the predict-api reload path)
    - With identical deterministic input, two consecutive runs produce
      numerically equal X_reduced arrays (IncrementalPCA is deterministic)

Key invariants checked :
  - X_reduced.shape == (n_samples, actual_dim)  dtype == float32
  - No NaN values in the output feature matrix
  - pca_image.pkl and pca_text.pkl files are valid sklearn objects

Dependencies : train-api/services/preprocess_text.py + pca_reducer.py (no TF)
               conftest fixtures: preprocess_text_module, pca_reducer_module
"""
import pytest
import numpy as np
import pandas as pd
import os
import pickle


@pytest.mark.integration
class TestTextToPCAMiniPipeline:
    def test_full_text_pipeline(self, tmp_path, preprocess_text_module,
                                pca_reducer_module):
        """extract_text_features → reduce_features produces a usable X_reduced."""
        df = pd.DataFrame({
            "designation": ["product"] * 5,
            "description": [
                "leather handbag premium quality",
                "running shoes breathable mesh",
                "winter jacket insulated",
                "smart phone 128gb storage",
                "bluetooth earphones noise cancel",
            ],
            "imageid":   list(range(5)),
            "productid": list(range(5)),
        })

        # 1. Extract text features
        text_features, vectorizer = preprocess_text_module.extract_text_features(df)
        assert text_features.shape[0] == 5
        np.save(tmp_path / "text.npy", text_features)

        # 2. Fake image features (ResNet50 output dim)
        image_features = np.random.rand(5, 2048).astype(np.float32)
        np.save(tmp_path / "image.npy", image_features)

        # 3. PCA reduction
        # n_text_components = min(target_dim - n_img, batch_size, n_samples, n_features)
        # With 5 samples, batch_size=5, n_img=3: n_text = min(7, 5, 5, ...) = 5 → total = 8
        x_path, pca_img_path, pca_txt_path = pca_reducer_module.reduce_features(
            text_features_path=str(tmp_path / "text.npy"),
            image_features_path=str(tmp_path / "image.npy"),
            n_components_img=3,
            target_dim=10,
            save_dir=str(tmp_path),
            initial_batch_size=5,
        )

        # 4. Assertions — actual columns depend on min(batch_size, n_samples)
        X = np.load(x_path)
        assert X.shape[0] == 5, f"Expected 5 rows, got {X.shape[0]}"
        assert X.shape[1] >= 3, f"Expected at least 3 columns (img PCA), got {X.shape[1]}"
        assert X.dtype == np.float32
        assert not np.isnan(X).any()
        assert os.path.exists(pca_img_path)
        assert os.path.exists(pca_txt_path)

    def test_pca_objects_can_transform_new_samples(self, tmp_path,
                                                    preprocess_text_module,
                                                    pca_reducer_module):
        """Saved PCA objects can transform unseen samples at inference time."""
        df = pd.DataFrame({"designation": ["product"]*10, "description": ["book paperback fiction"]*10, "imageid": list(range(10)), "productid": list(range(10))})
        text_features, _ = preprocess_text_module.extract_text_features(df)
        np.save(tmp_path / "t.npy", text_features)
        img = np.random.rand(10, 2048).astype(np.float32)
        np.save(tmp_path / "i.npy", img)

        _, pca_img_path, pca_txt_path = pca_reducer_module.reduce_features(
            text_features_path=str(tmp_path / "t.npy"),
            image_features_path=str(tmp_path / "i.npy"),
            n_components_img=3,
            target_dim=8,
            save_dir=str(tmp_path),
            initial_batch_size=5,
        )

        with open(pca_img_path, "rb") as f:
            pca_img = pickle.load(f)
        with open(pca_txt_path, "rb") as f:
            pca_txt = pickle.load(f)

        # Transform a single new sample
        new_img  = np.random.rand(1, 2048).astype(np.float32)
        new_text = np.random.rand(1, text_features.shape[1]).astype(np.float32)

        reduced_img  = pca_img.transform(new_img)
        reduced_text = pca_txt.transform(new_text)

        assert reduced_img.shape[0]  == 1
        assert reduced_text.shape[0] == 1

    def test_deterministic_output_same_seed(self, tmp_path, preprocess_text_module,
                                             pca_reducer_module):
        """Same data → same X_reduced (IncrementalPCA is deterministic)."""
        df = pd.DataFrame({"designation": ["product"]*6, "description": ["apple orange grape"]*6, "imageid": list(range(6)), "productid": list(range(6))})
        text_features, _ = preprocess_text_module.extract_text_features(df)

        for run in range(2):
            np.save(tmp_path / "t.npy", text_features)
            img = np.ones((6, 2048), dtype=np.float32)
            np.save(tmp_path / "i.npy", img)
            out_dir = tmp_path / f"run{run}"
            out_dir.mkdir()
            x_path, _, _ = pca_reducer_module.reduce_features(
                text_features_path=str(tmp_path / "t.npy"),
                image_features_path=str(tmp_path / "i.npy"),
                n_components_img=2,
                target_dim=6,
                save_dir=str(out_dir),
                initial_batch_size=3,
            )
            if run == 0:
                X0 = np.load(x_path)
            else:
                X1 = np.load(x_path)

        np.testing.assert_allclose(X0, X1, rtol=1e-4,
                                   err_msg="PCA output not deterministic")
