import os
import pickle
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import dagshub
import dagshub.auth
from dagshub import get_repo_bucket_client
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# --- Global config ---
ARTIFACTS_DIR = "artifacts"  # Changed to relative path for DVC
LOCAL_ARTIFACTS_DIR = "/app/data/artifacts"  # For Docker container
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LOCAL_ARTIFACTS_DIR, exist_ok=True)

MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "rakuten_z")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "rakuten_multimodal")

# --- DagsHub / MLflow token setup ---
DAGSHUB_USER = os.getenv("DAGSHUB_USER")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_CACHE = os.getenv("DAGSHUB_CLIENT_TOKENS_CACHE")

def _acquire_dagshub_token():
    # Prefer explicit token from env
    if DAGSHUB_TOKEN:
        print("Using DAGSHUB_TOKEN from environment")
        try:
            if DAGSHUB_CACHE:
                dagshub.auth.add_app_token(DAGSHUB_TOKEN, cache_location=DAGSHUB_CACHE)
            else:
                dagshub.auth.add_app_token(DAGSHUB_TOKEN)
            print("Stored DAGSHUB_TOKEN in token cache via add_app_token()")
        except Exception as e:
            print(f"Warning: failed to add token to cache: {e}")
        return DAGSHUB_TOKEN

    # Try token cache
    try:
        if DAGSHUB_CACHE:
            token = dagshub.auth.get_token(fail_if_no_token=True, cache_location=DAGSHUB_CACHE)
        else:
            token = dagshub.auth.get_token(fail_if_no_token=True)
        print("Loaded Dagshub token from cache")
        return token
    except RuntimeError:
        print("No Dagshub token found in cache (fail_if_no_token=True). Skipping OAuth.")
        return None
    except Exception as e:
        print(f"Warning: error while attempting to load Dagshub token from cache: {e}")
        return None

_token = _acquire_dagshub_token()

# --- Initialize Dagshub ---
if _token and DAGSHUB_USER:
    os.environ["DAGSHUB_TOKEN"] = _token
    try:
        dagshub.init(repo_owner=DAGSHUB_USER, repo_name="rakuten_z", mlflow=True)
        print("DagsHub initialized successfully")
    except Exception as e:
        print(f"Warning: Dagshub init failed: {e}")

# --- MLflow tracking URI ---
if _token and DAGSHUB_USER:
    # URL-style authentication
    MLFLOW_TRACKING_URI = f"https://{DAGSHUB_USER}:{_token}@dagshub.com/{DAGSHUB_USER}/rakuten_z.mlflow"
    # Also set environment variables for S3 access
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://dagshub.com"
    os.environ["AWS_ACCESS_KEY_ID"] = DAGSHUB_USER
    os.environ["AWS_SECRET_ACCESS_KEY"] = _token
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
else:
    # fallback, no token
    MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USER}/rakuten_z.mlflow" if DAGSHUB_USER else "http://localhost:5000"

print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI.split('@')[-1] if '@' in MLFLOW_TRACKING_URI else MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Experiment setup ---
try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"MLflow experiment set to '{MLFLOW_EXPERIMENT}'")
except Exception as e:
    print(f"Warning: could not set MLflow experiment: {e}")

# Disable autolog — manual logging only
mlflow.tensorflow.autolog(disable=True)

# --- Helper functions ---
def evaluate_model(model, X, y, label_encoder):
    y_encoded = label_encoder.transform(y)
    probs = model.predict(X)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_encoded, y_pred)
    report = classification_report(y_encoded, y_pred, output_dict=True)

    # Log metrics
    try:
        mlflow.log_metric("final_accuracy", float(acc))
        for label, metrics in report.items():
            if isinstance(metrics, dict) and "f1-score" in metrics:
                key = f"f1_{label}"
                mlflow.log_metric(key, float(np.asarray(metrics["f1-score"]).item()))
    except Exception as e:
        print(f"Warning: failed to log final metrics: {e}")

    return {"accuracy": acc, "report": report}

def _to_python_native(obj):
    if isinstance(obj, dict):
        return {k: _to_python_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj

def save_training_history(history, artifacts_dir="/app/data/artifacts"):
    """Save training history in format expected by Airflow DAG"""
    history_path = os.path.join(artifacts_dir, "train_history.json")
    
    # Convert numpy types to Python native for JSON serialization
    history_serializable = {}
    for key, values in history.history.items():
        history_serializable[key] = [float(val) for val in values]
    
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    print(f"Training history saved to: {history_path}")
    return history_path

# --- Training function ---
def _log_datasets(train_data, x_csv_path, y_csv_path, X_reduced):
    """Log source CSVs and reduced feature matrix to the active MLflow run."""
    try:
        cols = [c for c in ["imageid", "productid", "description", "prdtypecode"] if c in train_data.columns]
        src_dataset = mlflow.data.from_pandas(
            train_data[cols],
            source=x_csv_path or "X_train_update.csv",
            name="rakuten_train",
            targets="prdtypecode",
        )
        mlflow.log_input(src_dataset, context="training")
        print(f"Logged dataset 'rakuten_train': {len(train_data)} rows, source={x_csv_path}")
    except Exception as e:
        print(f"Warning: failed to log source dataset: {e}")

    try:
        reduced_dataset = mlflow.data.from_numpy(
            X_reduced,
            source=x_csv_path or "X_train_update.csv",
            name="X_reduced_features",
        )
        mlflow.log_input(reduced_dataset, context="training")
        print(f"Logged dataset 'X_reduced_features': shape={X_reduced.shape}")
    except Exception as e:
        print(f"Warning: failed to log reduced features dataset: {e}")


def build_and_train_model(
    X, y,
    epochs=10, batch_size=32, run_name="training_run",
    pca_models=None,
    train_data=None, x_csv_path=None, y_csv_path=None,
):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    inputs = Input(shape=(X.shape[1],))
    x = Dense(512, activation="relu")(inputs)
    x = Dropout(0.5)(x)
    outputs = Dense(len(label_encoder.classes_), activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Repo bucket client optional
    try:
        _ = get_repo_bucket_client(f"{DAGSHUB_USER}/rakuten_z", flavor="boto")
        print("Repo bucket client obtained")
    except Exception as e:
        print(f"Warning: could not obtain repo bucket client: {e}")

    # End any active run
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception as e:
        print(f"Warning: error ending active run: {e}")

    with mlflow.start_run(run_name=run_name) as run:
        # Log params
        try:
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("input_dim", X.shape[1])
            mlflow.log_param("num_classes", len(label_encoder.classes_))
            mlflow.log_param("dataset_rows", int(len(X)))
            if pca_models:
                pca_img = pca_models.get("image")
                pca_txt = pca_models.get("text")
                if pca_img:
                    # Use sklearn's standard param names so existing DagsHub columns are filled
                    mlflow.log_param("copy", pca_img.copy)
                    mlflow.log_param("n_components", pca_img.n_components_)
                    mlflow.log_param("pca_text_n_components", pca_txt.n_components_ if pca_txt else None)
        except Exception as e:
            print(f"Warning: failed to log params: {e}")

        # Log datasets (source CSVs + reduced feature matrix)
        if train_data is not None:
            _log_datasets(train_data, x_csv_path, y_csv_path, X)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Log metrics per epoch
        for epoch in range(len(history.history.get("loss", []))):
            mlflow.log_metric("train_loss", float(history.history["loss"][epoch]), step=epoch)
            if "val_loss" in history.history:
                mlflow.log_metric("val_loss", float(history.history["val_loss"][epoch]), step=epoch)
            if "accuracy" in history.history:
                mlflow.log_metric("train_accuracy", float(history.history["accuracy"][epoch]), step=epoch)
            if "val_accuracy" in history.history:
                mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][epoch]), step=epoch)

        # Save in .keras format (Keras 3 native) — avoids legacy H5 deserialisation issues
        model_path_dvc = os.path.join(ARTIFACTS_DIR, "neural_network_model.keras")
        model_path_local = os.path.join(LOCAL_ARTIFACTS_DIR, "neural_network_model.keras")

        print(f"Saving model to DVC path: {model_path_dvc}")
        print(f"Saving model to local path: {model_path_local}")

        model.save(model_path_dvc)
        model.save(model_path_local)
        
        encoder_path_dvc = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
        encoder_path_local = os.path.join(LOCAL_ARTIFACTS_DIR, "label_encoder.pkl")
        
        with open(encoder_path_dvc, "wb") as f:
            pickle.dump(label_encoder, f)
        with open(encoder_path_local, "wb") as f:
            pickle.dump(label_encoder, f)

        # ✅ ENHANCEMENT: Save training history for Airflow DAG
        history_path_dvc = os.path.join(ARTIFACTS_DIR, "train_history.json")
        history_path_local = save_training_history(history, LOCAL_ARTIFACTS_DIR)
        
        # Also save to DVC artifacts directory
        history_serializable = {}
        for key, values in history.history.items():
            history_serializable[key] = [float(val) for val in values]
        
        with open(history_path_dvc, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"Training history saved to DVC path: {history_path_dvc}")

        # Log artifacts to MLflow
        try:
            mlflow.log_artifact(model_path_dvc, artifact_path="artifacts")
            mlflow.log_artifact(encoder_path_dvc, artifact_path="artifacts")
            mlflow.log_artifact(history_path_dvc, artifact_path="artifacts")
            print("Artifacts logged to MLflow")
        except Exception as e:
            print(f"Warning: failed to log artifacts to MLflow: {e}")

        # Log PCA sklearn models so DagsHub shows their params
        if pca_models:
            try:
                if pca_models.get("image"):
                    mlflow.sklearn.log_model(pca_models["image"], artifact_path="pca_image")
                if pca_models.get("text"):
                    mlflow.sklearn.log_model(pca_models["text"], artifact_path="pca_text")
                print("PCA models logged to MLflow")
            except Exception as e:
                print(f"Warning: failed to log PCA models: {e}")

        # Log model with signature to MLflow
        try:
            preds_for_signature = model.predict(X_train[: min(64, len(X_train))])
            signature = infer_signature(X_train[: min(64, len(X_train))], preds_for_signature)
            mlflow.tensorflow.log_model(model, artifact_path="model", signature=signature)
            print("Model logged to MLflow")
        except Exception as e:
            print(f"Warning: mlflow.tensorflow.log_model failed: {e}")

        # Evaluate and log final metrics
        eval_results = evaluate_model(model, X, y, label_encoder)

        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

    # Register model in MLflow Model Registry
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    registered_version_number = None
    try:
        try:
            client.create_registered_model(MLFLOW_MODEL_NAME)
            print(f"Created registered model: {MLFLOW_MODEL_NAME}")
        except Exception:
            print(f"Registered model {MLFLOW_MODEL_NAME} already exists")
        
        model_uri = f"runs:/{run_id}/model"
        registered_version = client.create_model_version(
            name=MLFLOW_MODEL_NAME,
            source=model_uri,
            run_id=run_id
        )
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=registered_version.version,
            stage="Staging",
            archive_existing_versions=False
        )
        registered_version_number = registered_version.version
        print(f"Model registered successfully: {MLFLOW_MODEL_NAME} v{registered_version.version}")
    except Exception as e:
        print(f"Warning: failed to register model: {e}")

    # Close run
    try:
        mlflow.end_run()
    except Exception:
        pass

    if not os.path.exists(model_path_dvc):
        raise FileNotFoundError(f"Model file not created at DVC path: {model_path_dvc}")
    if not os.path.exists(encoder_path_dvc):
        raise FileNotFoundError(f"Encoder file not created at DVC path: {encoder_path_dvc}")
    if not os.path.exists(history_path_dvc):
        raise FileNotFoundError(f"History file not created at DVC path: {history_path_dvc}")
    
    print(f"✅ Model saved locally at: {model_path_dvc}")
    print(f"✅ Model saved locally at: {model_path_local}")
    print(f"✅ Training history saved at: {history_path_local}")
    print(f"✅ Model logged to DagsHub MLflow: {MLFLOW_TRACKING_URI}")

    return model, label_encoder, history, model_path_dvc, run_id, registered_version_number, eval_results