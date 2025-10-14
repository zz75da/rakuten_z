import os
import pickle
import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Depends, Header
from prometheus_client import make_asgi_app, Counter, Histogram
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import CountVectorizer
import base64
import cv2
from pydantic import BaseModel

# --- FastAPI app ---
app = FastAPI(title="Prediction API", version="1.0")

# --- Prometheus metrics ---
REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total prediction requests",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "predict_request_latency_seconds", "Prediction request latency", ["endpoint"]
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
    global model, label_encoder, text_vectorizer, pca_text, pca_image
    try:
        model = tf.keras.models.load_model(
            os.path.join(ARTIFACTS_PATH, "neural_network_model.h5")
        )
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
        print("Artifacts loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")


# --- Preprocessing helpers ---
def extract_text_features(description: str):
    processed = [" ".join(description.lower().split())]
    bow = text_vectorizer.transform(processed).toarray()
    reduced = pca_text.transform(bow)
    return reduced


def extract_image_features(image_input: str):
    """
    Accepts either:
    - a path to a .npy file
    - a base64-encoded image string
    """
    # 1️ Check if input is a valid .npy path
    if os.path.exists(image_input) and image_input.endswith(".npy"):
        try:
            features = np.load(image_input)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to load .npy features: {e}"
            )
    else:
        # 2️ Assume base64 image
        try:
            image_bytes = base64.b64decode(image_input)
            img_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Image decode failed")
            img_flat = img.flatten()[None, :]
            features = img_flat
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")
    # Apply PCA
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


@app.post("/predict-text")
def predict_text(req: TextRequest, user: dict = Depends(verify_jwt_token)):
    text_features = extract_text_features(req.description)
    dummy_img = np.zeros((1, pca_image.n_components_))
    combined = np.hstack([text_features, dummy_img])

    probs = model.predict(combined)
    pred = int(np.argmax(probs, axis=1)[0])
    label = str(label_encoder.inverse_transform([pred])[0])

    REQUEST_COUNT.labels("/predict-text", "POST", "200").inc()
    return {
        "pred_class": pred,
        "label": label,
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

    REQUEST_COUNT.labels("/predict-image", "POST", "200").inc()
    return {
        "pred_class": pred,
        "label": label,
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

    REQUEST_COUNT.labels("/predict-multimodal", "POST", "200").inc()
    return {
        "pred_class": pred,
        "label": label,
        "probs": probs.tolist(),
        "mode": "multimodal",
    }


# --- Prometheus metrics ---
prometheus_app = make_asgi_app()
app.mount("/metrics", prometheus_app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5003)
