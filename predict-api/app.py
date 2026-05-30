import os
import pickle
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import json
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
import cv2
from pydantic import BaseModel
from services.drift_monitor import record_prediction, trigger_report, buffer_size, reference_exists
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
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH") or "/app/data/artifacts"
GATE_API_URL = os.getenv("GATE_API_URL", "http://gate-api:5000")

# --- Globals ---
# TF-IDF vectorizer model (upgraded from CountVectorizer — same .transform() API)
model_cv = None
text_vectorizer: TfidfVectorizer = None
pca_text: IncrementalPCA = None
# MiniLM-L12 model (paraphrase-multilingual-MiniLM-L12-v2, 384-d)
model_minilm = None
minilm_encoder = None
# mpnet model (paraphrase-multilingual-mpnet-base-v2, 768-d)
model_mpnet  = None
mpnet_encoder = None
# CLIP ViT-B/32 model
model_clip     = None
clip_tokenizer = None
clip_text_model = None
# Shared
label_encoder: LabelEncoder = None
pca_image: IncrementalPCA = None
resnet_model = None
_gradcam_model = None   # lazily built from resnet_model on first GradCAM request

# Encoder model names — must match what was used during training / encoding
_MINILM_MODEL_NAME = os.getenv("MINILM_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_MPNET_MODEL_NAME  = os.getenv("MPNET_MODEL",  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

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
    model: str = "cv"  # "cv" | "minilm" | "clip"


class ImageRequest(BaseModel):
    image_base64: str


class MultimodalRequest(BaseModel):
    description: str = None
    image_base64: str = None
    model: str = "cv"  # "cv" | "minilm" | "clip"


class BatchStreamRequest(BaseModel):
    items: list[MultimodalRequest]
    batch_size: int = 50


class GradCAMRequest(BaseModel):
    image_base64: str
    model: str = "clip"       # which trained model to use for class prediction
    target_class: int = None  # if None, uses predicted class


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
    global model_cv, model_minilm, model_mpnet, model_clip, label_encoder, text_vectorizer, \
           pca_text, pca_image, resnet_model, minilm_encoder, mpnet_encoder, \
           clip_tokenizer, clip_text_model
    # Late fusion models use Lambda layers — enable unsafe deserialization for our own trusted models
    try:
        import keras
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass
    try:
        # --- Shared artifacts ---
        label_encoder = pickle.load(open(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"), "rb"))
        pca_image = pickle.load(open(os.path.join(ARTIFACTS_PATH, "pca_image.pkl"), "rb"))

        # --- CV model ---
        try:
            keras_path = os.path.join(ARTIFACTS_PATH, "neural_network_model.keras")
            h5_path = os.path.join(ARTIFACTS_PATH, "neural_network_model.h5")
            model_path = keras_path if os.path.exists(keras_path) else h5_path
            model_cv = tf.keras.models.load_model(model_path, compile=False)
            text_vectorizer = pickle.load(open(os.path.join(ARTIFACTS_PATH, "text_vectorizer.pkl"), "rb"))
            pca_text = pickle.load(open(os.path.join(ARTIFACTS_PATH, "pca_text.pkl"), "rb"))
            print("✓ CV model loaded from disk")
        except Exception as e:
            print(f"⚠ CV model could not be loaded: {e}")
            model_cv = None

        # --- MiniLM model (optional — only available after a minilm training run) ---
        minilm_keras = os.path.join(ARTIFACTS_PATH, "neural_network_model_minilm.keras")
        if os.path.exists(minilm_keras):
            try:
                model_minilm = tf.keras.models.load_model(minilm_keras, compile=False)
                # Load encoder directly — no pkl, model cached by sentence-transformers
                from sentence_transformers import SentenceTransformer
                minilm_encoder = SentenceTransformer(_MINILM_MODEL_NAME, device="cpu")
                print("✓ MiniLM model loaded")
            except Exception as e:
                print(f"⚠ MiniLM model could not be loaded: {e}")
                model_minilm = None
                minilm_encoder = None
        else:
            print("ℹ MiniLM model not found — run minilm-encoder then train with text_encoder=minilm")
            model_minilm = None
            minilm_encoder = None

        # --- mpnet model (optional — only available after an mpnet training run) ---
        mpnet_keras = os.path.join(ARTIFACTS_PATH, "neural_network_model_mpnet.keras")
        if os.path.exists(mpnet_keras):
            try:
                model_mpnet = tf.keras.models.load_model(mpnet_keras, compile=False)
                from sentence_transformers import SentenceTransformer
                mpnet_encoder = SentenceTransformer(_MPNET_MODEL_NAME, device="cpu")
                print(f"✓ mpnet model loaded ({_MPNET_MODEL_NAME})")
            except Exception as e:
                print(f"⚠ mpnet model could not be loaded: {e}")
                model_mpnet  = None
                mpnet_encoder = None
        else:
            print("ℹ mpnet model not found — run minilm-encoder then train with text_encoder=mpnet")
            model_mpnet  = None
            mpnet_encoder = None

        # --- CLIP model (optional — only available after a clip training run) ---
        clip_keras = os.path.join(ARTIFACTS_PATH, "neural_network_model_clip.keras")
        if os.path.exists(clip_keras):
            try:
                model_clip = tf.keras.models.load_model(clip_keras, compile=False)
                from transformers import CLIPTokenizer, CLIPTextModel
                clip_tokenizer  = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
                clip_text_model.eval()
                print("✓ CLIP model loaded")
            except Exception as e:
                print(f"⚠ CLIP model could not be loaded: {e}")
                model_clip = None
                clip_tokenizer = None
                clip_text_model = None
        else:
            print("ℹ CLIP model not found — run clip-encoder then train with text_encoder=clip")
            model_clip = None
            clip_tokenizer = None
            clip_text_model = None

        # --- ResNet50 for image features (shared) ---
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


def extract_text_features_minilm(description: str):
    embeddings = minilm_encoder.encode([description], convert_to_numpy=True)
    return embeddings.astype(np.float32)  # shape (1, 384)


def extract_text_features_mpnet(description: str):
    embeddings = mpnet_encoder.encode([description], convert_to_numpy=True)
    return embeddings.astype(np.float32)  # shape (1, 768)


def extract_text_features_clip(description: str):
    import torch
    inputs = clip_tokenizer(
        [description], padding=True, truncation=True, max_length=77, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = clip_text_model(**inputs)
        emb = outputs.pooler_output                              # (1, 512)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)        # L2-normalise
    return emb.cpu().numpy().astype(np.float32)


def _resolve_model(model_key: str):
    """Return (model, text_dim) for the requested encoder, or raise 503."""
    if model_key == "minilm":
        if model_minilm is None:
            raise HTTPException(
                status_code=503,
                detail="MiniLM model not available — trigger the DAG to train it first",
            )
        return model_minilm, 384
    if model_key == "mpnet":
        if model_mpnet is None:
            raise HTTPException(
                status_code=503,
                detail="mpnet model not available — trigger the DAG to train it first",
            )
        return model_mpnet, 768
    if model_key == "clip":
        if model_clip is None:
            raise HTTPException(
                status_code=503,
                detail="CLIP model not available — trigger the DAG to train it first",
            )
        return model_clip, 512
    if model_cv is None:
        raise HTTPException(status_code=503, detail="CV model not available")
    if pca_text is None:
        raise HTTPException(
            status_code=503,
            detail="CV model artifacts incomplete — pca_text.pkl missing",
        )
    return model_cv, int(pca_text.n_components_)


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


# --- Drift monitoring endpoints ---
@app.get("/drift-rebuild-reference-info")
def drift_rebuild_reference_info(_user: dict = Depends(verify_jwt_token)):
    """Reference is built by train-api POST /drift-rebuild-reference (has CSV access)."""
    return {
        "info": "Call POST /drift-rebuild-reference on train-api (port 5002) to rebuild.",
        "reference_exists": reference_exists(),
    }


@app.post("/drift-trigger-report")
def drift_trigger_report(user: dict = Depends(verify_jwt_token)):
    """Force a drift report from the current prediction buffer. Admin only."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    n = buffer_size()
    trigger_report()
    return {"status": "report_triggered", "buffer_size_at_trigger": n}


@app.get("/drift-status")
def drift_status(_user: dict = Depends(verify_jwt_token)):
    """Return current drift monitoring status."""
    from pathlib import Path
    report_dir = Path("/app/data/artifacts/drift_reports")
    reports = sorted(report_dir.glob("drift_*.html"), key=lambda p: p.stat().st_mtime) if report_dir.exists() else []
    return {
        "reference_exists":   reference_exists(),
        "buffer_size":        buffer_size(),
        "buffer_capacity":    2000,
        "n_reports_on_disk":  len(reports),
        "latest_report":      reports[-1].name if reports else None,
    }


# --- Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "cv_model_loaded":     model_cv     is not None,
        "minilm_model_loaded": model_minilm is not None,
        "mpnet_model_loaded":  model_mpnet  is not None,
        "clip_model_loaded":   model_clip   is not None,
    }


@app.post("/reload-artifacts")
def reload_artifacts_endpoint():
    """Reload all artifacts from disk — call after a new training run completes."""
    try:
        load_artifacts()
        # Guard: at minimum one model must be loaded
        if model_cv is None and model_minilm is None and model_mpnet is None and model_clip is None:
            raise RuntimeError("No model could be loaded from disk")
        if model_cv is not None and pca_text is None:
            raise RuntimeError("CV model loaded but pca_text.pkl is missing")
        return {
            "status": "reloaded",
            "cv_model_loaded":     model_cv     is not None,
            "minilm_model_loaded": model_minilm is not None,
            "mpnet_model_loaded":  model_mpnet  is not None,
            "clip_model_loaded":   model_clip   is not None,
            "pca_text_components":  int(pca_text.n_components_) if pca_text else None,
            "pca_image_components": int(pca_image.n_components_),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifact reload failed: {e}")


@app.post("/predict-text")
def predict_text(req: TextRequest, user: dict = Depends(verify_jwt_token)):
    active_model, text_dim = _resolve_model(req.model)
    if req.model == "minilm":
        text_features = extract_text_features_minilm(req.description)
    elif req.model == "mpnet":
        text_features = extract_text_features_mpnet(req.description)
    elif req.model == "clip":
        text_features = extract_text_features_clip(req.description)
    else:
        text_features = extract_text_features(req.description)
    dummy_img = np.zeros((1, pca_image.n_components_))
    combined = np.hstack([text_features, dummy_img])

    probs = active_model.predict(combined)
    pred = int(np.argmax(probs, axis=1)[0])
    label = str(label_encoder.inverse_transform([pred])[0])
    category = CATEGORY_NAMES.get(label, label)

    _record_drift_metrics(probs[0], label, "/predict-text")
    FEATURE_TEXT_MEAN.set(float(np.mean(text_features)))
    REQUEST_COUNT.labels("/predict-text", "POST", "200").inc()
    return {
        "pred_class": pred, "label": label, "category": category,
        "probs": probs.tolist(), "mode": "text_only", "encoder": req.model,
    }


@app.post("/predict-image")
def predict_image(req: ImageRequest, user: dict = Depends(verify_jwt_token)):
    active_model, text_dim = _resolve_model("cv")  # image-only always uses CV model
    image_features = extract_image_features(req.image_base64)
    dummy_text = np.zeros((1, text_dim))
    combined = np.hstack([dummy_text, image_features])

    probs = active_model.predict(combined)
    pred = int(np.argmax(probs, axis=1)[0])
    label = str(label_encoder.inverse_transform([pred])[0])
    category = CATEGORY_NAMES.get(label, label)

    _record_drift_metrics(probs[0], label, "/predict-image")
    FEATURE_IMAGE_MEAN.set(float(np.mean(image_features)))
    REQUEST_COUNT.labels("/predict-image", "POST", "200").inc()
    return {
        "pred_class": pred, "label": label, "category": category,
        "probs": probs.tolist(), "mode": "image_only", "encoder": "cv",
    }


@app.post("/predict-multimodal")
def predict_multimodal(req: MultimodalRequest, user: dict = Depends(verify_jwt_token)):
    if not req.description and not req.image_base64:
        raise HTTPException(status_code=400, detail="Must provide description or image_base64")

    active_model, text_dim = _resolve_model(req.model)

    # ── Text features (fallback: zeros if missing) ────────────────────────────
    has_text = bool(req.description and req.description.strip())
    if has_text:
        try:
            if req.model == "minilm":
                text_features = extract_text_features_minilm(req.description)
            elif req.model == "mpnet":
                text_features = extract_text_features_mpnet(req.description)
            elif req.model == "clip":
                text_features = extract_text_features_clip(req.description)
            else:
                text_features = extract_text_features(req.description)
        except Exception as e:
            print(f"Text extraction failed, falling back to zeros: {e}")
            text_features = np.zeros((1, text_dim), dtype=np.float32)
            has_text = False
    else:
        text_features = np.zeros((1, text_dim), dtype=np.float32)

    # ── Image features (fallback: zeros if missing or decode error) ───────────
    has_image = bool(req.image_base64)
    if has_image:
        try:
            image_features = extract_image_features(req.image_base64)
        except Exception as e:
            print(f"Image extraction failed, falling back to zeros: {e}")
            image_features = np.zeros((1, pca_image.n_components_), dtype=np.float32)
            has_image = False
    else:
        image_features = np.zeros((1, pca_image.n_components_), dtype=np.float32)

    # ── Determine effective mode ──────────────────────────────────────────────
    if has_text and has_image:
        mode = "multimodal"
    elif has_text:
        mode = "text_only_fallback"
    else:
        mode = "image_only_fallback"

    combined = np.hstack([text_features, image_features])
    probs    = active_model.predict(combined)
    pred     = int(np.argmax(probs, axis=1)[0])
    label    = str(label_encoder.inverse_transform([pred])[0])
    category = CATEGORY_NAMES.get(label, label)
    confidence = float(np.max(probs[0]))

    _record_drift_metrics(probs[0], label, "/predict-multimodal")
    FEATURE_TEXT_MEAN.set(float(np.mean(text_features)))
    FEATURE_IMAGE_MEAN.set(float(np.mean(image_features)))
    REQUEST_COUNT.labels("/predict-multimodal", "POST", "200").inc()
    # Record lightweight summary stats for drift monitoring (no raw high-dim vectors)
    try:
        record_prediction({
            "prdtypecode": label,
            "confidence":  round(confidence, 4),
            "text_norm":   round(float(np.linalg.norm(text_features)), 4),
            "image_norm":  round(float(np.linalg.norm(image_features)), 4),
            "encoder":     req.model,
            "mode":        mode,
        })
    except Exception:
        pass

    response = {
        "pred_class": pred, "label": label, "category": category,
        "probs": probs.tolist(), "mode": mode, "encoder": req.model,
    }
    # Warn caller when operating in fallback mode with low confidence
    if mode != "multimodal" and confidence < 0.4:
        response["warning"] = (
            f"Operating in {mode} — confidence is low ({confidence:.2f}). "
            f"Provide both text and image for best results."
        )
    return response


def _get_gradcam_model():
    """
    Lazily build a sub-model that outputs (last_conv_layer, global_avg_pool)
    from the already-loaded resnet_model.  No extra weights loaded.
    Cached in _gradcam_model so it's only built once.
    """
    global _gradcam_model
    if resnet_model is None:
        return None
    if _gradcam_model is None:
        try:
            last_conv = resnet_model.get_layer("conv5_block3_out")
            _gradcam_model = tf.keras.Model(
                inputs=resnet_model.input,
                outputs=[last_conv.output, resnet_model.output],
            )
        except Exception as e:
            print(f"GradCAM model build failed: {e}")
            return None
    return _gradcam_model


def _compute_gradcam(img_rgb: np.ndarray, active_model, target_class: int = None) -> np.ndarray:
    """
    Class-aware GradCAM for our PCA-in-the-middle pipeline.

    Because PCA is non-differentiable, we can't backprop through the full
    image → ResNet → PCA → Dense chain.  Instead we use an analytical
    approximation:
      1. Compute ResNet50 last-conv activations (7×7×2048).
      2. Project the dense head's image-branch weights back to ResNet50 space
         via the PCA components matrix, giving a 2048-d class vector.
      3. Weight the spatial activation map by that vector → class-discriminative
         heatmap without requiring gradient flow through PCA.

    Falls back to plain mean-activation (no class guidance) if model introspection
    fails (e.g., architecture mismatch).
    """
    gm = _get_gradcam_model()
    if gm is None:
        raise HTTPException(status_code=503, detail="GradCAM model not available")

    img_batch = resnet_preprocess(
        np.expand_dims(img_rgb.astype(np.float32), axis=0)
    )

    conv_out, pool_out = gm(img_batch, training=False)
    conv_out = conv_out.numpy()[0]   # (7, 7, 2048)
    pool_out = pool_out.numpy()      # (1, 2048)

    # ── Determine target class from full pipeline if not specified ────────────
    if target_class is None and active_model is not None and pca_image is not None:
        try:
            img_pca = pca_image.transform(pool_out)   # (1, 384)
            # Zero-pad text dim so shape matches model input
            input_dim = active_model.input_shape[1]
            text_dim  = input_dim - img_pca.shape[1]
            combined  = np.hstack([np.zeros((1, text_dim)), img_pca])
            probs     = active_model.predict(combined, verbose=0)
            target_class = int(np.argmax(probs[0]))
        except Exception:
            target_class = None   # fall back to mean activation

    # ── Class-discriminative weights in ResNet50 space ────────────────────────
    # Strategy: find the image-branch output Dense layer by name (late fusion)
    # or by position (early fusion), project class weights back through PCA.
    # Falls back to mean activation (no class guidance) on any failure.
    cam_weights = None
    if target_class is not None and active_model is not None and pca_image is not None:
        try:
            n_classes_local = len(label_encoder.classes_) if label_encoder else 0
            n_img = pca_image.n_components_

            # Prefer the named image_logits layer (late fusion model)
            img_logits_layer = None
            for layer in active_model.layers:
                if layer.name == "image_logits" and len(layer.get_weights()) == 2:
                    img_logits_layer = layer
                    break

            if img_logits_layer is not None:
                # Late fusion: image_logits Dense(n_img_hidden → n_classes)
                # Find the preceding image Dense layer
                img_out_w = img_logits_layer.get_weights()[0]  # (hidden, n_classes)
                class_w = img_out_w[:, target_class]            # (hidden,)
                img_dense = None
                for layer in active_model.layers:
                    if layer.name == "image_dense1" and len(layer.get_weights()) == 2:
                        img_dense = layer
                        break
                if img_dense is not None:
                    img_d_w = img_dense.get_weights()[0]   # (n_img, hidden)
                    resnet_w = img_d_w.T @ pca_image.components_  # (hidden, 2048)
                    cam_weights = (resnet_w @ class_w).ravel()     # (2048,)
            else:
                # Early fusion: find last Dense(n_classes) and preceding Dense
                all_dense = [
                    l for l in active_model.layers
                    if len(l.get_weights()) == 2
                    and l.get_weights()[0].shape[-1] == n_classes_local
                ]
                if all_dense:
                    out_w = all_dense[-1].get_weights()[0]  # (hidden_2, n_classes)
                    class_w = out_w[:, target_class]         # (hidden_2,)
                    # Find Dense before this one (hidden_1 × hidden_2)
                    for layer in reversed(active_model.layers):
                        if (len(layer.get_weights()) == 2
                                and layer.get_weights()[0].shape[-1] == len(class_w)
                                and layer is not all_dense[-1]):
                            prev_w = layer.get_weights()[0]   # (input_dim, hidden_2)
                            img_w  = prev_w[-n_img:, :]        # (n_img, hidden_2)
                            resnet_w = img_w.T @ pca_image.components_
                            cam_weights = (resnet_w @ class_w).ravel()
                            break
        except Exception as e:
            print(f"GradCAM weight projection failed ({e}), using mean activation")

    # ── Build spatial heatmap ─────────────────────────────────────────────────
    if cam_weights is not None and len(cam_weights) == conv_out.shape[2]:
        heatmap = np.maximum(conv_out @ cam_weights, 0)           # (7, 7)
    else:
        heatmap = np.mean(conv_out, axis=2)                        # (7, 7) fallback

    heatmap = heatmap / (np.max(np.abs(heatmap)) + 1e-8)
    return heatmap, target_class


@app.post("/gradcam")
def gradcam_endpoint(req: GradCAMRequest, user: dict = Depends(verify_jwt_token)):
    """
    Generate a class-discriminative GradCAM heatmap for an uploaded image.
    Returns the heatmap overlaid on the original image as base64 JPEG.

    Memory: ~50 MB peak (gradient-free — uses analytical weight projection).
    Disk  : nothing saved — result returned inline.
    """
    if resnet_model is None or pca_image is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    try:
        image_bytes = base64.b64decode(req.image_base64)
        img_array   = np.frombuffer(image_bytes, np.uint8)
        img_bgr     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Image decode failed")
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")

    try:
        active_model, _ = _resolve_model(req.model)
    except HTTPException:
        active_model = None

    heatmap, used_class = _compute_gradcam(img_rgb, active_model, req.target_class)

    # ── Overlay on original image ─────────────────────────────────────────────
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color   = cv2.applyColorMap(
        np.uint8(255 * np.clip(heatmap_resized, 0, 1)), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6,
        heatmap_color, 0.4, 0,
    )

    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    heatmap_b64 = base64.b64encode(buf).decode("utf-8")

    label = str(label_encoder.inverse_transform([used_class])[0]) if (
        used_class is not None and label_encoder is not None
    ) else "unknown"

    REQUEST_COUNT.labels("/gradcam", "POST", "200").inc()
    return {
        "heatmap_base64": heatmap_b64,
        "predicted_class": used_class,
        "predicted_label": label,
        "encoder": req.model,
    }


@app.post("/predict-multimodal-batch-stream")
def predict_multimodal_batch_stream(req: BatchStreamRequest, user: dict = Depends(verify_jwt_token)):
    def generate():
        for item in req.items:
            try:
                active_model, text_dim = _resolve_model(item.model)
                if item.description:
                    if item.model == "minilm":
                        text_features = extract_text_features_minilm(item.description)
                    elif item.model == "clip":
                        text_features = extract_text_features_clip(item.description)
                    else:
                        text_features = extract_text_features(item.description)
                else:
                    text_features = np.zeros((1, text_dim))
                image_features = (
                    extract_image_features(item.image_base64)
                    if item.image_base64
                    else np.zeros((1, pca_image.n_components_))
                )
                combined = np.hstack([text_features, image_features])
                probs = active_model.predict(combined, verbose=0)
                pred  = int(np.argmax(probs, axis=1)[0])
                label = str(label_encoder.inverse_transform([pred])[0])
                result = {
                    "pred_class": pred,
                    "label":      label,
                    "category":   CATEGORY_NAMES.get(label, label),
                    "probs":      probs.tolist(),
                    "encoder":    item.model,
                }
                _record_drift_metrics(probs[0], label, "/predict-multimodal-batch-stream")
                REQUEST_COUNT.labels("/predict-multimodal-batch-stream", "POST", "200").inc()
            except Exception as e:
                result = {"error": str(e)}
                REQUEST_COUNT.labels("/predict-multimodal-batch-stream", "POST", "500").inc()
            yield json.dumps(result) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# --- Prometheus metrics ---
prometheus_app = make_asgi_app()
app.mount("/metrics", prometheus_app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5003)
