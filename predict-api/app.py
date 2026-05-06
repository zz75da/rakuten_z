import os
import pickle
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from fastapi import FastAPI, HTTPException, Depends, Header
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import CountVectorizer
import base64
import cv2
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient

# --- FastAPI app ---
app = FastAPI(title="Prediction API", version="1.0")

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "rakuten_multimodal")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Prometheus metrics ---
REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total prediction requests",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "predict_request_latency_seconds",
    "Prediction request latency",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# --- Drift detection metrics ---
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence (max softmax probability) of each prediction",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
PREDICTION_ENTROPY = Histogram(
    "prediction_entropy",
    "Shannon entropy of prediction probabilities (high = uncertain = possible drift)",
    ["endpoint"],
    buckets=[0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.5],
)
PREDICTION_CLASS_COUNT = Counter(
    "prediction_class_total",
    "Predictions per class label (tracks class distribution drift)",
    ["endpoint", "label"],
)
FEATURE_TEXT_MEAN = Gauge(
    "feature_text_input_mean",
    "Mean of the text feature vector for the last prediction",
)
FEATURE_IMAGE_MEAN = Gauge(
    "feature_image_input_mean",
    "Mean of the image feature vector for the last prediction",
)

# --- Configuration ---
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "/app/data/artifacts")
GATE_API_URL = os.getenv("GATE_API_URL", "http://gate-api:5000")

# --- Globals ---
model = None
label_encoder: LabelEncoder = None
text_vectorizer: CountVectorizer = None
pca_text: IncrementalPCA = None
pca_image: IncrementalPCA = None
resnet_model = None          # loaded once at startup for image feature extraction

# --- Human-readable category names for each Rakuten prdtypecode ---
CATEGORY_NAMES: dict[str, str] = {
    "10":   "Books",
    "40":   "Movies & DVDs",
    "50":   "Video Game Accessories",
    "60":   "Handheld & Retro Consoles",
    "1140": "Collectible Figures & Novelty Decor",
    "1160": "Trading Cards & Card Games",
    "1180": "Tabletop Gaming Miniatures",
    "1280": "Toys & Puzzles",
    "1281": "Children's Games & Educational Toys",
    "1300": "RC Vehicles & Drones",
    "1301": "Sports & Outdoor Games",
    "1302": "Novelty Toys & Leisure",
    "1320": "Baby & Nursery Products",
    "1560": "Furniture & Storage",
    "1920": "Home Textiles & Soft Furnishings",
    "1940": "Food, Beverages & Grocery",
    "2060": "Lighting & Home Decoration",
    "2220": "Pet Supplies",
    "2280": "Magazines & Periodicals",
    "2403": "Manga & Comics",
    "2462": "Video Game Consoles & Electronics",
    "2522": "Office & Art Supplies",
    "2582": "Outdoor & Garden Furniture",
    "2583": "Swimming Pool & Spa Equipment",
    "2585": "Gardening Tools & Equipment",
    "2705": "History & Documentary Media",
    "2905": "Digital Games & Software",
}

# --- Input Schemas ---
class TextRequest(BaseModel):
    description: str


class ImageRequest(BaseModel):
    image_base64: str  # image in base64


class MultimodalRequest(BaseModel):
    description: str = None
    image_base64: str = None


# --- Auth helper ---
def verify_jwt_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]
    try:
        resp = requests.post(
            f"{GATE_API_URL}/validate-token",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        user_data = resp.json()
        user_data["token"] = token
        return user_data
    except requests.RequestException:
        raise HTTPException(
            status_code=503, detail="Cannot reach gate-api for token validation"
        )


# --- Load artifacts ---
def load_artifacts():
    global model, label_encoder, text_vectorizer, pca_text, pca_image, resnet_model
    try:
        # Try loading from MLflow first
        try:
            print(f"Attempting to load model '{MLFLOW_MODEL_NAME}' from MLflow ({MLFLOW_TRACKING_URI})...")
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
            model = mlflow.tensorflow.load_model(model_uri)
            print(f"✓ Model loaded from MLflow: {model_uri}")
            
            # Load supporting artifacts from MLflow (if available as logged artifacts)
            label_encoder = pickle.load(open(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"), "rb"))
            text_vectorizer = pickle.load(open(os.path.join(ARTIFACTS_PATH, "text_vectorizer.pkl"), "rb"))
            pca_text = pickle.load(open(os.path.join(ARTIFACTS_PATH, "pca_text.pkl"), "rb"))
            pca_image = pickle.load(open(os.path.join(ARTIFACTS_PATH, "pca_image.pkl"), "rb"))
            print("✓ Supporting artifacts loaded from disk")
            
        except Exception as mlflow_error:
            print(f"MLflow loading failed, falling back to disk artifacts: {mlflow_error}")
            # Prefer .keras (Keras 3 native), fall back to .h5 for older artifacts
            keras_path = os.path.join(ARTIFACTS_PATH, "neural_network_model.keras")
            h5_path = os.path.join(ARTIFACTS_PATH, "neural_network_model.h5")
            model_path = keras_path if os.path.exists(keras_path) else h5_path
            model = tf.keras.models.load_model(model_path)
            label_encoder = pickle.load(
                open(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"), "rb")
            )
            text_vectorizer = pickle.load(
                open(os.path.join(ARTIFACTS_PATH, "text_vectorizer.pkl"), "rb")
            )
            pca_text = pickle.load(open(os.path.join(ARTIFACTS_PATH, "pca_text.pkl"), "rb"))
            pca_image = pickle.load(
                open(os.path.join(ARTIFACTS_PATH, "pca_image.pkl"), "rb")
            )
            print("✓ Artifacts loaded from disk")

        # ResNet50 for image feature extraction (shared across both load paths)
        if resnet_model is None:
            print("Loading ResNet50 for image feature extraction...")
            resnet_model = ResNet50(
                weights="imagenet", include_top=False,
                pooling="avg", input_shape=(224, 224, 3)
            )
            print("✓ ResNet50 loaded")

    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")


# --- Drift helpers ---
def _record_drift_metrics(probs_flat: np.ndarray, label: str, endpoint: str):
    confidence = float(np.max(probs_flat))
    entropy = float(-np.sum(probs_flat * np.log(probs_flat + 1e-10)))
    PREDICTION_CONFIDENCE.labels(endpoint=endpoint).observe(confidence)
    PREDICTION_ENTROPY.labels(endpoint=endpoint).observe(entropy)
    PREDICTION_CLASS_COUNT.labels(endpoint=endpoint, label=label).inc()


# --- Preprocessing helpers ---
def extract_text_features(description: str):
    processed = [" ".join(description.lower().split())]
    bow = text_vectorizer.transform(processed).toarray()
    reduced = pca_text.transform(bow)
    return reduced


def extract_image_features(image_input: str):
    """
    Accepts either:
    - a path to a pre-computed .npy feature file (shape: [1, 2048])
    - a base64-encoded JPEG/PNG image string

    In the base64 case the image is passed through ResNet50
    (weights=imagenet, pooling=avg) to produce a 2048-d embedding,
    matching the feature space used during training.
    """
    if os.path.exists(image_input) and image_input.endswith(".npy"):
        try:
            features = np.load(image_input)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to load .npy features: {e}"
            )
    else:
        try:
            image_bytes = base64.b64decode(image_input)
            img_array = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Image decode failed — invalid or corrupted image data")
            # Resize to 224×224 and convert BGR→RGB (OpenCV loads BGR)
            img_rgb = cv2.cvtColor(
                cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB
            )
            # Add batch dimension and apply ResNet50 preprocessing
            img_batch = resnet_preprocess(
                np.expand_dims(img_rgb.astype(np.float32), axis=0)
            )
            # Extract 2048-d global-average-pooled features
            features = resnet_model.predict(img_batch, verbose=0)  # shape (1, 2048)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")

    try:
        reduced = pca_image.transform(features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PCA transform failed: {e}")
    return reduced


# --- Startup ---
@app.on_event("startup")
def startup_event():
    load_artifacts()


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reload-artifacts")
def reload_artifacts_endpoint():
    """Reload all artifacts from disk — call after a new training run completes."""
    try:
        load_artifacts()
        return {
            "status": "reloaded",
            "pca_text_components": int(pca_text.n_components_),
            "pca_image_components": int(pca_image.n_components_),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifact reload failed: {e}")


@app.post("/predict-text")
def predict_text(req: TextRequest, user: dict = Depends(verify_jwt_token)):
    text_features = extract_text_features(req.description)
    dummy_img = np.zeros((1, pca_image.n_components_))
    combined = np.hstack([text_features, dummy_img])

    probs = model.predict(combined)
    pred = int(np.argmax(probs, axis=1)[0])
    label = str(label_encoder.inverse_transform([pred])[0])
    category = CATEGORY_NAMES.get(label, label)

    _record_drift_metrics(probs[0], label, "/predict-text")
    FEATURE_TEXT_MEAN.set(float(np.mean(text_features)))
    REQUEST_COUNT.labels("/predict-text", "POST", "200").inc()
    return {
        "pred_class": pred,
        "label": label,
        "category": category,
        "probs": probs.tolist(),
        "mode": "text_only",
    }


@app.post("/predict-image")
def predict_image(req: ImageRequest, user: dict = Depends(verify_jwt_token)):
    image_features = extract_image_features(req.image_base64)
    dummy_text = np.zeros((1, pca_text.n_components_))
    combined = np.hstack([dummy_text, image_features])

    probs = model.predict(combined)
    pred = int(np.argmax(probs, axis=1)[0])
    label = str(label_encoder.inverse_transform([pred])[0])
    category = CATEGORY_NAMES.get(label, label)

    _record_drift_metrics(probs[0], label, "/predict-image")
    FEATURE_IMAGE_MEAN.set(float(np.mean(image_features)))
    REQUEST_COUNT.labels("/predict-image", "POST", "200").inc()
    return {
        "pred_class": pred,
        "label": label,
        "category": category,
        "probs": probs.tolist(),
        "mode": "image_only",
    }


@app.post("/predict-multimodal")
def predict_multimodal(req: MultimodalRequest, user: dict = Depends(verify_jwt_token)):
    if not req.description and not req.image_base64:
        raise HTTPException(
            status_code=400, detail="Must provide description or image_base64"
        )

    text_features = (
        extract_text_features(req.description)
        if req.description
        else np.zeros((1, pca_text.n_components_))
    )
    image_features = (
        extract_image_features(req.image_base64)
        if req.image_base64
        else np.zeros((1, pca_image.n_components_))
    )

    combined = np.hstack([text_features, image_features])

    probs = model.predict(combined)
    pred = int(np.argmax(probs, axis=1)[0])
    label = str(label_encoder.inverse_transform([pred])[0])
    category = CATEGORY_NAMES.get(label, label)

    _record_drift_metrics(probs[0], label, "/predict-multimodal")
    FEATURE_TEXT_MEAN.set(float(np.mean(text_features)))
    FEATURE_IMAGE_MEAN.set(float(np.mean(image_features)))
    REQUEST_COUNT.labels("/predict-multimodal", "POST", "200").inc()
    return {
        "pred_class": pred,
        "label": label,
        "category": category,
        "probs": probs.tolist(),
        "mode": "multimodal",
    }


# --- Prometheus metrics ---
prometheus_app = make_asgi_app()
app.mount("/metrics", prometheus_app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5003)
