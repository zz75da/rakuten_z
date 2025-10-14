# streamlit/app_streamlit.py - MULTIPAGE STREAMLIT + MLflow
import streamlit as st
import requests
import json
import os
import base64
from pathlib import Path
import pandas as pd
from mlflow.tracking import MlflowClient

# --- API URLs ---
GATE_API_URL = os.environ.get("GATE_API_URL", "http://gate-api:5000/login")
PREDICT_API_URL = os.environ.get("PREDICT_API_URL", "http://predict-api:5003")

# --- MLflow setup ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "rakuten_model")
#  You can set your preferred default here (e.g., "Production" or "6")
DEFAULT_MODEL_STAGE_OR_VERSION = os.environ.get("DEFAULT_MODEL_VERSION", "6")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# ======================
# UTILS
# ======================
@st.cache_data(ttl=300, show_spinner=False)
def list_model_versions_and_stages(model_name: str):
    """Fetch all versions and stages for a given model from MLflow."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        stages = set(["Production", "Staging"])
        version_numbers = sorted({v.version for v in versions}, key=lambda x: int(x))
        return stages.union(version_numbers)
    except Exception as e:
        st.warning(f"⚠️ Could not retrieve MLflow versions: {e}")
        return {"Production", "Staging"}


def get_model_uri(stage_or_version="Production"):
    """Return the full model URI (source or models:/...) for a given stage or version."""
    try:
        if str(stage_or_version).isdigit():
            version_info = client.get_model_version(name=MODEL_NAME, version=str(stage_or_version))
            return f"models:/{MODEL_NAME}/{version_info.version}", version_info.version, version_info.current_stage or "None"
        else:
            versions = client.get_latest_versions(MODEL_NAME, stages=[stage_or_version])
            if versions:
                v = versions[0]
                return f"models:/{MODEL_NAME}/{stage_or_version}", v.version, v.current_stage
            else:
                st.warning(f"⚠️ No model found in MLflow for stage '{stage_or_version}'.")
                return None, None, None
    except Exception as e:
        st.error(f"Failed to fetch model from MLflow: {e}")
        return None, None, None


def authenticate_user(username, password):
    try:
        response = requests.post(GATE_API_URL, json={"username": username, "password": password})
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to authentication service.")
        return None


def predict_single(description, uploaded_image, token, model_uri):
    payload = {}
    if description:
        payload["description"] = description
    if uploaded_image:
        image_bytes = uploaded_image.getvalue()
        payload["image_features"] = base64.b64encode(image_bytes).decode("utf-8")  # Predict-API expects image_features key in base64
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Choisir l'endpoint selon ce qui est fourni
    if uploaded_image and description:
        endpoint = f"{PREDICT_API_URL}/predict-multimodal"
    elif uploaded_image:
        endpoint = f"{PREDICT_API_URL}/predict-image"
    else:
        endpoint = f"{PREDICT_API_URL}/predict-text"

    try:
        with st.spinner("Making prediction..."):
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"Predicted Category: {result['label']} (class {result['pred_class']})")
            if "probs" in result:
                st.write("Probabilities per class:", result["probs"])
            if uploaded_image:
                st.image(uploaded_image, caption="Uploaded image", use_column_width=True)
        else:
            st.error(f"API Error: {resp.text}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


def predict_batch_stream(batch_items, token, output_file_path, model_uri, chunk_size=50):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with output_file_path.open("w", encoding="utf-8") as f_out:
        total_items = len(batch_items)
        st.info(f"Streaming predictions for {total_items} items in chunks of {chunk_size}...")

        for i in range(0, total_items, chunk_size):
            chunk = batch_items[i:i+chunk_size]
            payload = {"items": chunk, "batch_size": chunk_size}

            try:
                with requests.post(f"{PREDICT_API_URL}/predict-multimodal-batch-stream",
                                   json=payload, headers=headers, stream=True, timeout=600) as resp:
                    if resp.status_code == 200:
                        for line in resp.iter_lines():
                            if line:
                                decoded = line.decode("utf-8")
                                f_out.write(decoded + "\n")
                                f_out.flush()
                                st.text(f"✅ Chunk {i//chunk_size + 1}/{(total_items-1)//chunk_size + 1}: {decoded}")
                    else:
                        st.error(f"API error on chunk {i//chunk_size + 1}: {resp.text}")
            except Exception as e:
                st.error(f"Chunk {i//chunk_size + 1} failed: {e}")

    st.success(f"🎉 All batch predictions saved to {output_file_path}")


# ======================
# PAGES
# ======================
def show_presentation_page():
    st.title("Présentation")
    st.markdown("""
    ### Objectifs
    - Classification multimodale (texte + image)
    - Mise en place d’un pipeline **MLOps** complet
    - Déploiement multi-services via **Docker Compose**
    """)
    st.markdown("### ⚙️ Architecture globale")
    st.code("""
Utilisateur -> Streamlit UI -> Gate-API (auth)
                       |
                 Predict-API -> MLflow Registry
    """, language="text")
    st.markdown("### Pipeline ML multimodal")
    st.code("""
Texte : TF-IDF -> PCA
Image : ResNet50 -> PCA
         |
         v
   Concaténation -> Dense -> Softmax -> Classe prédite
    """, language="text")


def show_prediction_page():
    st.title("MLOps Streamlit Prediction Interface")

    # Login
    if "user_token" not in st.session_state:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            auth_resp = authenticate_user(username, password)
            if auth_resp:
                st.session_state["user_token"] = auth_resp["token"]
                st.success("Login successful!")
            else:
                st.error("Login failed")

    if "user_token" in st.session_state:
        # --- Model selection ---
        st.sidebar.subheader("Select MLflow Model Version")
        available = list_model_versions_and_stages(MODEL_NAME)
        available_sorted = sorted(available, key=lambda x: (not str(x).isdigit(), x))  # numeric first

        # Default selection
        default_index = available_sorted.index(DEFAULT_MODEL_STAGE_OR_VERSION) if DEFAULT_MODEL_STAGE_OR_VERSION in available_sorted else 0
        stage_or_version = st.sidebar.selectbox("Stage or Version", available_sorted, index=default_index)

        model_uri, version_num, current_stage = get_model_uri(str(stage_or_version))
        if not model_uri:
            st.stop()

        st.sidebar.markdown(f" **Using model:** `{MODEL_NAME}`")
        st.sidebar.markdown(f" **URI:** `{model_uri}`")
        st.sidebar.markdown(f" **Version:** `{version_num}`  |  **Stage:** `{current_stage}`")

        # --- Prediction forms ---
        st.subheader("Single Prediction")
        description = st.text_input("Product description")
        uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        if st.button("Predict Single"):
            predict_single(description, uploaded_image, st.session_state["user_token"], model_uri)

        # --- Batch Prediction ---
        st.subheader("Batch Prediction (Streaming)")
        uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
        descriptions_file = st.file_uploader("Optional: CSV with descriptions", type=["csv"])
        if st.button("Predict Batch"):
            if uploaded_files:
                batch_items = []
                descriptions = []
                if descriptions_file:
                    df_desc = pd.read_csv(descriptions_file)
                    if "description" in df_desc.columns:
                        descriptions = df_desc["description"].tolist()
                for idx, file in enumerate(uploaded_files):
                    desc = descriptions[idx] if idx < len(descriptions) else ""
                    batch_items.append({
                        "description": desc,
                        "image_features": base64.b64encode(file.getvalue()).decode("utf-8")
                    })
                output_file = "./batch_predictions_streamed.jsonl"
                predict_batch_stream(batch_items, st.session_state["user_token"], output_file, model_uri)
            else:
                st.warning("⚠️ Please upload at least one image")



def show_docker_workflow():
    st.title("Docker Compose Workflow")
    st.markdown("### Aperçu: micro-services orchestrés via Docker Compose")
    st.code("""
Utilisateur ─▶ Streamlit UI
                      │
                      ▼
               Gate-API (auth)
                      │
                      ▼
             Predict-API (inférence)
                      │
                      ▼
               MLflow (gestion modèle)
    """, language="text")


# ======================
# MAIN
# ======================
st.sidebar.title("Menu")
page = st.sidebar.radio("Navigation", ["Présentation du projet", "Prédictions", "Workflow Docker"])

if page == "Présentation du projet":
    show_presentation_page()
elif page == "Prédictions":
    show_prediction_page()
elif page == "Workflow Docker":
    show_docker_workflow()
