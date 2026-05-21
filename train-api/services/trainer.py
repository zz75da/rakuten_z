import os
import gc
import pickle
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

load_dotenv()

ARTIFACTS_DIR     = "/app/data/artifacts"


def _load_params() -> dict:
    """Load params.yaml; fall back to empty dict so hardcoded defaults still work."""
    for path in ["/app/params.yaml", "params.yaml"]:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f) or {}
    return {}


_PARAMS = _load_params()
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "rakuten_z")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "rakuten_multimodal")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Read env vars at module level — no network calls here
_DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
_DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
_DAGSHUB_CACHE = os.getenv("DAGSHUB_CLIENT_TOKENS_CACHE")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_python_native(obj):
    """Recursively convert numpy scalars/arrays to plain Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_python_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_training_history(history, artifacts_dir=ARTIFACTS_DIR, text_encoder="countvectorizer"):
    if text_encoder == "minilm":
        filename = "train_history_minilm.json"
    elif text_encoder == "clip":
        filename = "train_history_clip.json"
    else:
        filename = "train_history.json"
    path = os.path.join(artifacts_dir, filename)
    with open(path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"Training history saved: {path}")
    return path


def evaluate_model(model, X_val, y_val_encoded, label_encoder):
    """Evaluate on the held-out validation set.

    Accepts pre-encoded y so the full X_reduced can be freed before this call.
    Returns plain-Python dict (no numpy types) ready for JSON serialisation.
    """
    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_val_encoded, y_pred)
    report = classification_report(y_val_encoded, y_pred, output_dict=True)

    try:
        mlflow.log_metric("final_val_accuracy", float(acc))
        for label, metrics in report.items():
            if isinstance(metrics, dict) and "f1-score" in metrics:
                mlflow.log_metric(f"f1_{label}", float(np.asarray(metrics["f1-score"]).item()))
    except Exception as e:
        print(f"Warning: failed to log final metrics: {e}")

    return _to_python_native({"accuracy": acc, "report": report})


def _log_datasets(train_data, x_csv_path, y_csv_path, X_reduced):
    try:
        cols = [c for c in ["imageid", "productid", "description", "prdtypecode"] if c in train_data.columns]
        mlflow.log_input(
            mlflow.data.from_pandas(train_data[cols], source=x_csv_path or "X_train_update.csv",
                                    name="rakuten_train", targets="prdtypecode"),
            context="training",
        )
        print(f"Logged dataset 'rakuten_train': {len(train_data)} rows")
    except Exception as e:
        print(f"Warning: failed to log source dataset: {e}")

    try:
        mlflow.log_input(
            mlflow.data.from_numpy(X_reduced, source=x_csv_path or "X_train_update.csv",
                                   name="X_reduced_features"),
            context="training",
        )
        print(f"Logged 'X_reduced_features': shape={X_reduced.shape}")
    except Exception as e:
        print(f"Warning: failed to log reduced features: {e}")


def _init_mlflow():
    """Acquire DagsHub credentials and configure MLflow tracking.

    Called once at the start of build_and_train_model — never at import time —
    so subprocess startup does not block on network I/O.
    Returns the tracking URI string.
    """
    import dagshub
    import dagshub.auth

    token = _DAGSHUB_TOKEN
    if token:
        try:
            kw = {"cache_location": _DAGSHUB_CACHE} if _DAGSHUB_CACHE else {}
            dagshub.auth.add_app_token(token, **kw)
            print("Stored DAGSHUB_TOKEN in token cache")
        except Exception as e:
            print(f"Warning: failed to add token to cache: {e}")
    else:
        try:
            kw = {"cache_location": _DAGSHUB_CACHE} if _DAGSHUB_CACHE else {}
            token = dagshub.auth.get_token(fail_if_no_token=True, **kw)
            print("Loaded Dagshub token from cache")
        except RuntimeError:
            print("No Dagshub token found — MLflow will use local fallback")
        except Exception as e:
            print(f"Warning: error loading Dagshub token: {e}")

    if token and _DAGSHUB_USER:
        os.environ["DAGSHUB_TOKEN"] = token
        try:
            dagshub.init(repo_owner=_DAGSHUB_USER, repo_name="rakuten_z", mlflow=True)
            print("DagsHub initialized successfully")
        except Exception as e:
            print(f"Warning: Dagshub init failed: {e}")
        tracking_uri = (
            f"https://{_DAGSHUB_USER}:{token}@dagshub.com/{_DAGSHUB_USER}/rakuten_z.mlflow"
        )
        os.environ.update({
            "MLFLOW_S3_ENDPOINT_URL": "https://dagshub.com",
            "AWS_ACCESS_KEY_ID":       _DAGSHUB_USER,
            "AWS_SECRET_ACCESS_KEY":   token,
            "AWS_DEFAULT_REGION":      "us-east-1",
        })
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        print(f"MLflow experiment set to '{MLFLOW_EXPERIMENT}'")
    except Exception as e:
        print(f"Warning: could not set MLflow experiment: {e}")

    return tracking_uri


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def build_and_train_model(
    X, y,
    epochs=10, batch_size=32, run_name="training_run",
    pca_models=None,
    train_data=None, x_csv_path=None, y_csv_path=None,
    text_encoder="countvectorizer", use_dev_images=False,
):
    # Network I/O happens here, not at import time
    tracking_uri = _init_mlflow()

    # Deferred TF imports (CPU image: no BFCAllocator, safe after any numpy work)
    import tensorflow as tf
    import mlflow.tensorflow
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2 as keras_l2
    mlflow.tensorflow.autolog(disable=True)

    # Architecture / training constants — read from params.yaml, fall back to defaults
    _m = _PARAMS.get("model", {})
    _t = _PARAMS.get("train", {})
    RANDOM_SEED   = _t.get("random_seed", 42)
    VAL_SPLIT     = 0.2
    LEARNING_RATE = _m.get("learning_rate", 0.001)
    HIDDEN_1      = _m.get("hidden_1", 512)
    HIDDEN_2      = _m.get("hidden_2", 256)
    DROPOUT_1     = _m.get("dropout_1", 0.3)
    DROPOUT_2     = _m.get("dropout_2", 0.2)
    L2_REG        = float(_m.get("l2_reg", 0.0))
    ES_PATIENCE   = _m.get("early_stopping_patience", 5)
    LR_PATIENCE   = _m.get("lr_patience", 2)
    LR_FACTOR     = _m.get("lr_factor", 0.3)
    LR_MIN        = 1e-6
    _reg          = keras_l2(L2_REG) if L2_REG > 0 else None

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Capture shape and class count before freeing X
    input_dim = X.shape[1]
    n_classes  = len(label_encoder.classes_)

    # End any stale MLflow run before starting ours
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

    try:
        mlflow.enable_system_metrics_logging()
    except Exception:
        pass

    with mlflow.start_run(run_name=run_name) as run:

        # Log datasets while X is still in scope
        if train_data is not None:
            _log_datasets(train_data, x_csv_path, y_csv_path, X)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        # Free the full feature matrix (~2 GB) before model.fit to avoid OOM.
        # X_train and X_val are independent copies; input_dim is already captured.
        del X
        gc.collect()

        # Build model
        inp = Input(shape=(input_dim,))
        h = Dense(HIDDEN_1, activation="relu", kernel_regularizer=_reg)(inp)
        h = Dropout(DROPOUT_1)(h)
        h = Dense(HIDDEN_2, activation="relu", kernel_regularizer=_reg)(h)
        h = Dropout(DROPOUT_2)(h)
        out = Dense(n_classes, activation="softmax")(h)

        model = Model(inp, out)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Log all hyperparameters in one call
        _effective_model_name = (
            f"{MLFLOW_MODEL_NAME}_minilm" if text_encoder == "minilm"
            else f"{MLFLOW_MODEL_NAME}_clip" if text_encoder == "clip"
            else f"{MLFLOW_MODEL_NAME}_cv"
        )
        # Log full params.yaml in DVC dot-notation so DagsHub shows them as columns
        # (git is not available in train-api, so the commit-tag approach never worked)
        try:
            _dvc_params = {}
            for _section, _vals in _PARAMS.items():
                if isinstance(_vals, dict):
                    for _k, _v in _vals.items():
                        _dvc_params[f"{_section}.{_k}"] = _v
            if _dvc_params:
                mlflow.log_params(_dvc_params)
        except Exception:
            pass

        try:
            _p = _PARAMS.get("preprocess", {})
            params = {
                "text_encoder": text_encoder, "use_dev_images": use_dev_images,
                "epochs": epochs, "batch_size": batch_size,
                "random_seed": RANDOM_SEED, "val_split": VAL_SPLIT,
                "input_dim": input_dim, "num_classes": n_classes,
                "dataset_rows": int(len(y_encoded)),
                "learning_rate": LEARNING_RATE,
                "hidden_1": HIDDEN_1, "hidden_2": HIDDEN_2,
                "dropout_1": DROPOUT_1, "dropout_2": DROPOUT_2,
                "l2_reg": L2_REG,
                "class_weights": "balanced",
                "early_stopping_patience": ES_PATIENCE,
                "lr_reduce_patience": LR_PATIENCE,
                "lr_reduce_factor": LR_FACTOR, "lr_min": LR_MIN,
                "model_name": _effective_model_name,
                "pca_components": _p.get("pca_components", 300),
                "cv_max_features": _p.get("cv_max_features", 5000),
                "normalize_embeddings": _PARAMS.get("minilm", {}).get("normalize_embeddings", False),
            }
            if pca_models:
                pca_img = pca_models.get("image")
                pca_txt = pca_models.get("text")
                params["pca_image_components"] = pca_img.n_components_ if pca_img else "n/a"
                params["pca_text_components"]  = pca_txt.n_components_ if pca_txt else "n/a"
            mlflow.log_params(params)
        except Exception as e:
            print(f"Warning: failed to log params: {e}")

        from sklearn.utils.class_weight import compute_class_weight
        class_weight_dict = dict(enumerate(
            compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        ))

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[
                EarlyStopping(monitor="val_accuracy", patience=ES_PATIENCE,
                              restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR,
                                  patience=LR_PATIENCE, min_lr=LR_MIN, verbose=1),
            ],
            verbose=1,
        )

        actual_epochs = len(history.history.get("loss", []))
        try:
            mlflow.log_param("actual_epochs_trained", actual_epochs)
            # Batch all 4 metrics into one HTTP call per epoch (was 4 calls)
            for epoch in range(actual_epochs):
                epoch_metrics = {"train_loss": float(history.history["loss"][epoch])}
                if "val_loss"      in history.history:
                    epoch_metrics["val_loss"]      = float(history.history["val_loss"][epoch])
                if "accuracy"      in history.history:
                    epoch_metrics["train_accuracy"] = float(history.history["accuracy"][epoch])
                if "val_accuracy"  in history.history:
                    epoch_metrics["val_accuracy"]   = float(history.history["val_accuracy"][epoch])
                mlflow.log_metrics(epoch_metrics, step=epoch)
        except Exception as e:
            print(f"Warning: failed to log epoch metrics: {e}")

        # Save model — encoder-specific filename so no model overwrites another.
        if text_encoder == "minilm":
            model_filename = "neural_network_model_minilm.keras"
        elif text_encoder == "clip":
            model_filename = "neural_network_model_clip.keras"
        else:
            model_filename = "neural_network_model.keras"
        model_path = os.path.join(ARTIFACTS_DIR, model_filename)
        encoder_path = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
        model.save(model_path)
        with open(encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)
        history_path = save_training_history(history, text_encoder=text_encoder)
        print(f"Model saved: {model_path}")

        try:
            mlflow.log_artifact(model_path,   artifact_path="artifacts")
            mlflow.log_artifact(encoder_path, artifact_path="artifacts")
            mlflow.log_artifact(history_path, artifact_path="artifacts")
        except Exception as e:
            print(f"Warning: failed to log artifacts: {e}")

        if pca_models:
            try:
                if pca_models.get("image"):
                    mlflow.sklearn.log_model(pca_models["image"], artifact_path="pca_image")
                if pca_models.get("text"):
                    mlflow.sklearn.log_model(pca_models["text"],  artifact_path="pca_text")
            except Exception as e:
                print(f"Warning: failed to log PCA models: {e}")

        try:
            n_sample = min(64, len(X_train))
            sig = infer_signature(X_train[:n_sample], model.predict(X_train[:n_sample], verbose=0))
            mlflow.tensorflow.log_model(model, artifact_path="model", signature=sig)
        except Exception as e:
            print(f"Warning: mlflow.tensorflow.log_model failed: {e}")

        # Evaluate on held-out val set (X already freed; X_val still in scope)
        eval_results = evaluate_model(model, X_val, y_val, label_encoder)
        run_id = run.info.run_id

    # Register model in MLflow Model Registry
    _DESCRIPTIONS = {
        "minilm": (
            "Rakuten multimodal product classifier — MiniLM text encoder. "
            "Text: paraphrase-multilingual-MiniLM-L12-v2 (384-dim). "
            "Image: ResNet50 → IncrementalPCA(384). Dense 512→256→Dropout→27 classes."
        ),
        "clip": (
            "Rakuten multimodal product classifier — CLIP ViT-B/32 text encoder. "
            "Text: openai/clip-vit-base-patch32 CLIPTextModel (512-dim, L2-normalised). "
            "Image: ResNet50 → IncrementalPCA(384). Dense 512→256→Dropout→27 classes."
        ),
        "countvectorizer": (
            "Rakuten multimodal product classifier — CountVectorizer text encoder. "
            "Text: CountVectorizer(max_features=10000) → IncrementalPCA(512). "
            "Image: ResNet50 → IncrementalPCA(384). Dense 512→256→Dropout→27 classes."
        ),
    }
    _desc = _DESCRIPTIONS.get(text_encoder, "Rakuten multimodal product classifier")
    # Encoder-specific registry name so CV, CLIP, MiniLM appear as separate entries.
    if text_encoder == "minilm":
        effective_model_name = f"{MLFLOW_MODEL_NAME}_minilm"
    elif text_encoder == "clip":
        effective_model_name = f"{MLFLOW_MODEL_NAME}_clip"
    else:
        effective_model_name = f"{MLFLOW_MODEL_NAME}_cv"
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    registered_version_number = None
    try:
        try:
            client.create_registered_model(
                effective_model_name,
                description=_desc,
                tags={"encoder": text_encoder, "task": "product_classification",
                      "dataset": "rakuten_84916", "framework": "tensorflow/keras"},
            )
        except Exception:
            # Already exists — refresh description and tags
            try:
                client.update_registered_model(effective_model_name, description=_desc)
                for k, v in {"encoder": text_encoder, "task": "product_classification",
                             "dataset": "rakuten_84916", "framework": "tensorflow/keras"}.items():
                    client.set_registered_model_tag(effective_model_name, k, v)
            except Exception:
                pass
        registered_version = client.create_model_version(
            name=effective_model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
            description=f"encoder={text_encoder} | run_id={run_id}",
            tags={"encoder": text_encoder},
        )
        client.transition_model_version_stage(
            name=effective_model_name,
            version=registered_version.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        registered_version_number = registered_version.version
        print(f"Model registered: {effective_model_name} v{registered_version.version}")
    except Exception as e:
        print(f"Warning: failed to register model: {e}")

    try:
        mlflow.end_run()
    except Exception:
        pass

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not saved: {model_path}")

    return model, label_encoder, history, model_path, run_id, registered_version_number, eval_results
