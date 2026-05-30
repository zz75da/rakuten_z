import pickle
import os
from time import time

# Path to store all artifacts
ARTIFACTS_PATH = os.path.join("/app", "data", "artifacts")


# --- Utility: track execution time ---
def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"[Artifacts] Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper


@track_time
def save_artifacts(model, vectorizer, pca_models, label_encoder, skip_existing=True, text_encoder="countvectorizer"):
    """
    Save model, vectorizer, PCA objects, and label encoder to ARTIFACTS_PATH.

    Parameters:
        skip_existing (bool): if True, skip saving files that already exist.
        text_encoder (str): "countvectorizer" or "minilm" — determines artifact names.
    """
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    if text_encoder == "minilm":
        # No vectorizer/pca_text — predict-api loads SentenceTransformer directly
        artifacts_to_save = {
            "neural_network_model_minilm.keras": model,
            "pca_image.pkl": pca_models["image"],
            "label_encoder.pkl": label_encoder,
        }
    elif text_encoder == "mpnet":
        # No vectorizer/pca_text — predict-api loads SentenceTransformer directly
        # IMPORTANT: must NOT save text_vectorizer.pkl or pca_text.pkl here;
        # they are None for mpnet and would corrupt the CV TfidfVectorizer on disk.
        artifacts_to_save = {
            "neural_network_model_mpnet.keras": model,
            "pca_image.pkl": pca_models["image"],
            "label_encoder.pkl": label_encoder,
        }
    elif text_encoder == "clip":
        # No vectorizer/pca_text — predict-api loads CLIP model directly
        artifacts_to_save = {
            "neural_network_model_clip.keras": model,
            "pca_image.pkl": pca_models["image"],
            "label_encoder.pkl": label_encoder,
        }
    else:
        # countvectorizer — saves TF-IDF vectorizer and text PCA
        artifacts_to_save = {
            "neural_network_model.keras": model,
            "text_vectorizer.pkl": vectorizer,
            "pca_image.pkl": pca_models["image"],
            "pca_text.pkl": pca_models["text"],
            "label_encoder.pkl": label_encoder,
        }

    for filename, obj in artifacts_to_save.items():
        path = os.path.join(ARTIFACTS_PATH, filename)
        if skip_existing and os.path.exists(path):
            print(f"[Artifacts] Skipping existing file {path}")
            continue

        start_artifact_time = time()
        print(f"[Artifacts] Saving {filename} to {path} ...")

        try:
            if filename.endswith(".keras") or filename.endswith(".h5"):
                obj.save(path)
            else:
                with open(path, "wb") as f:
                    pickle.dump(obj, f)
            end_artifact_time = time()
            print(f"[Artifacts] Saved {filename} in {end_artifact_time - start_artifact_time:.2f} seconds")
        except Exception as e:
            print(f"[Artifacts] ERROR saving {filename}: {e}")

    print(f"[Artifacts] All artifacts processed successfully in {ARTIFACTS_PATH}")
