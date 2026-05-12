# streamlit/app_streamlit.py - MULTIPAGE STREAMLIT + MLflow
import streamlit as st
import requests
import json
import os
import re
import base64
from pathlib import Path
import pandas as pd
from mlflow.tracking import MlflowClient

# --- API URLs ---
GATE_API_URL = os.environ.get("GATE_API_URL", "http://gate-api:5000/login")
PREDICT_API_URL = os.environ.get("PREDICT_API_URL", "http://predict-api:5003")

# --- Test-set ground-truth lookup ---
# Loads the pre-classified test_data_zz.xlsx so predictions can be compared
# against the description-inferred expected category.
_TEST_DATA_PATH = os.environ.get(
    "TEST_DATA_PATH",
    "/app/data/test_data_zz.xlsx",        # inside container
)
_TEST_DATA_LOCAL = r"C:\Users\zobir\DScientest\ds_rakuren\test_data_zz.xlsx"

@st.cache_data(show_spinner=False)
def _load_test_lookup() -> dict:
    """Return {imageid_str: {category, prdtypecode, designation}} or {}."""
    for path in (_TEST_DATA_PATH, _TEST_DATA_LOCAL):
        try:
            df = pd.read_excel(path, engine="openpyxl",
                               usecols=["imageid", "designation",
                                        "prdtypecode", "category"])
            df["imageid"] = df["imageid"].astype(str).str.strip()
            return df.set_index("imageid").to_dict(orient="index")
        except Exception:
            continue
    return {}


def _lookup_expected(filename: str) -> dict | None:
    """
    Try to find an imageid in the test lookup from the uploaded filename.
    Rakuten test images are named  image_<imageid>_product_<productid>.jpg
    or simply  <imageid>.jpg – extract the first long numeric run.
    """
    lookup = _load_test_lookup()
    if not lookup:
        return None
    nums = re.findall(r"\d{6,}", filename)
    for n in nums:
        if n in lookup:
            return lookup[n]
    return None

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
        st.warning(f"Could not retrieve MLflow versions: {e}")
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
                st.warning(f"No model found in MLflow for stage '{stage_or_version}'.")
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
        payload["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    if uploaded_image and description:
        endpoint = f"{PREDICT_API_URL}/predict-multimodal"
    elif uploaded_image:
        endpoint = f"{PREDICT_API_URL}/predict-image"
    else:
        endpoint = f"{PREDICT_API_URL}/predict-text"

    # Try to find expected category from test-set lookup (by imageid in filename)
    expected = _lookup_expected(uploaded_image.name) if uploaded_image else None

    try:
        with st.spinner("Making prediction..."):
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            predicted_cat = result.get("category", result["label"])
            predicted_code = result["label"]

            # ── Result display ──────────────────────────────────────────────
            if expected:
                exp_cat  = expected.get("category", "?")
                exp_code = expected.get("prdtypecode", "?")
                match = predicted_code == str(exp_code)
                icon = "✅" if match else "⚠️"

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"<div style='background:#1e1e2e;border:2px solid #28a745;"
                        f"border-radius:8px;padding:14px;text-align:center'>"
                        f"<div style='color:#9aabb8;font-size:0.75rem'>MODEL PREDICTION</div>"
                        f"<div style='color:#28a745;font-weight:700;font-size:1rem;margin-top:4px'>"
                        f"{predicted_cat}</div>"
                        f"<div style='color:#6d7a9f;font-size:0.75rem'>code {predicted_code}</div>"
                        f"</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(
                        f"<div style='background:#1e1e2e;border:2px solid #a8b2d8;"
                        f"border-radius:8px;padding:14px;text-align:center'>"
                        f"<div style='color:#9aabb8;font-size:0.75rem'>EXPECTED (from description)</div>"
                        f"<div style='color:#a8b2d8;font-weight:700;font-size:1rem;margin-top:4px'>"
                        f"{exp_cat}</div>"
                        f"<div style='color:#6d7a9f;font-size:0.75rem'>code {exp_code}</div>"
                        f"</div>", unsafe_allow_html=True)

                verdict = (
                    f"{icon} **Match** — prediction aligns with description-based expectation."
                    if match else
                    f"{icon} **Mismatch** — model predicted **{predicted_cat}**, "
                    f"description suggests **{exp_cat}**. "
                    f"Check confidence: {100*max(result['probs'][0]):.0f}%"
                )
                st.caption(verdict)

            else:
                # No test-set entry — plain result
                st.success(f"**{predicted_cat}**  —  code `{predicted_code}` "
                           f"(confidence {100*max(result['probs'][0]):.0f}%)")

            if "probs" in result:
                with st.expander("Probabilities per class"):
                    st.write(result["probs"])
            if uploaded_image:
                st.image(uploaded_image, caption=uploaded_image.name,
                         use_column_width=True)
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
def _service_badge(name, url, path="/health"):
    try:
        r = requests.get(url + path, timeout=3)
        ok = r.status_code == 200
    except Exception:
        ok = False
    color = "#28a745" if ok else "#dc3545"
    status = "UP" if ok else "DOWN"
    style = "color:" + color + ";font-size:13px;font-weight:600"
    return "<span style=\"" + style + "\">● " + name + ": " + status + "</span>"


def show_presentation_page():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
                border-radius:12px;padding:36px 32px 28px;margin-bottom:28px">
        <h1 style="color:#e94560;margin:0 0 6px;font-size:2.2rem">
            Rakuten Product Classifier
        </h1>
        <p style="color:#a8b2d8;margin:0;font-size:1.05rem">
            Pipeline MLOps multimodal &mdash; texte &amp; image &mdash; du CSV au modèle en production
        </p>
        <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap">
            <span style="background:#e94560;color:white;padding:3px 10px;border-radius:20px;font-size:12px">Python 3.11</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">TensorFlow 2.17</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">MLflow + DagsHub</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">Airflow 2.x</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">Prometheus + Grafana</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">Docker Compose</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Classes produits",  "27",    help="Catégories Rakuten à prédire")
    c2.metric("Features texte",    "1 024", help="CountVectorizer(5000) → PCA(1024)")
    c3.metric("Features image",    "300",   help="ResNet50(2048) → PCA(300)")
    c4.metric("Features combinées","1 324", help="Concaténation texte + image")
    c5.metric("Val accuracy",      "~73.8 %", help="Dernier run complet (84 916 échantillons)")
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_ml, tab_ops, tab_stack, tab_services = st.tabs([
        "Pipeline ML", "Architecture MLOps", "Stack technique", "Statut des services"
    ])

    # ── Tab 1 : Pipeline ML ───────────────────────────────────────────────────
    with tab_ml:
        st.markdown("#### Extraction de features")
        col_txt, col_img = st.columns(2)

        with col_txt:
            st.markdown("""
            <div style="background:#1e1e2e;border-left:4px solid #e94560;
                        border-radius:8px;padding:18px 20px">
                <b style="color:#e94560">Branche Texte</b><br><br>
                <code style="color:#a8b2d8">description</code> <span style="color:#6d7a9f">(chaîne brute)</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;&nbsp;↓ SpaCy lemmatisation + stopwords</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;&nbsp;↓ CountVectorizer</span> <code>max_features=5000</code><br>
                &nbsp;&nbsp;&nbsp;↓ IncrementalPCA <code>n_components=1024</code><br>
                <b style="color:#28a745">→ vecteur 1 × 1024</b>
            </div>
            """, unsafe_allow_html=True)

        with col_img:
            st.markdown("""
            <div style="background:#1e1e2e;border-left:4px solid #0f3460;
                        border-radius:8px;padding:18px 20px">
                <b style="color:#a8b2d8">Branche Image</b><br><br>
                <code style="color:#a8b2d8">image_jpg</code> <span style="color:#6d7a9f">(224 × 224 px)</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;&nbsp;↓ ResNet50 pré-entraîné (ImageNet, sans top)</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;&nbsp;↓ Global Average Pooling →</span> <code>2048 dims</code><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;&nbsp;↓ IncrementalPCA</span> <code>n_components=300</code><br>
                <b style="color:#28a745">→ vecteur 1 × 300</b>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Modèle de classification")

        def _card(label, sub, border, label_color):
            return (
                f"<div style='background:#16213e;border:1px solid {border};"
                f"border-radius:6px;padding:6px 10px;font-size:0.78rem;"
                f"text-align:center;min-width:68px'>"
                f"<span style='color:{label_color};font-weight:600'>{label}</span><br>"
                f"<span style='color:#6d7a9f'>{sub}</span></div>"
            )

        arrow = "<div style='color:#6d7a9f;padding:0 5px;font-size:0.9rem;align-self:center'>&#9654;</div>"

        row = "".join([
            "<div style='display:flex;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:4px'>",
            # input boxes stacked
            "<div style='display:flex;flex-direction:column;gap:6px'>",
            f"<div style='background:#16213e;border:1px solid #e94560;border-radius:6px;"
            f"padding:6px 12px;font-size:0.78rem;text-align:center'>"
            f"<span style='color:#e94560;font-weight:600'>texte</span>"
            f"<span style='color:#6d7a9f'> 1&times;1024</span></div>",
            f"<div style='background:#16213e;border:1px solid #4a6fa5;border-radius:6px;"
            f"padding:6px 12px;font-size:0.78rem;text-align:center'>"
            f"<span style='color:#a8b2d8;font-weight:600'>image</span>"
            f"<span style='color:#6d7a9f'> 1&times;300</span></div>",
            "</div>",
            # merge bracket + hstack
            "<div style='color:#6d7a9f;font-size:1.6rem;padding:0 2px;align-self:center;line-height:0.9'>}</div>",
            f"<div style='background:#0f3460;border-radius:6px;padding:6px 10px;"
            f"font-size:0.78rem;color:#a8b2d8;text-align:center;min-width:60px;align-self:center'>"
            f"hstack<br><span style='color:#6d7a9f'>1&times;1324</span></div>",
            arrow,
            _card("Dense",   "512 &middot; ReLU",  "#28a745", "#28a745"),
            arrow,
            _card("Dropout", "p = 0.5",             "#9aabb8", "#a8b2d8"),
            arrow,
            _card("Dense",   "27 &middot; Softmax", "#e94560", "#e94560"),
            arrow,
            f"<div style='background:#1a3a1a;border:1px solid #28a745;border-radius:6px;"
            f"padding:6px 12px;font-size:0.82rem;text-align:center;min-width:56px;align-self:center'>"
            f"<span style='color:#28a745;font-weight:600'>classe</span><br>"
            f"<span style='color:#6d7a9f'>1 / 27</span></div>",
            "</div>",
        ])

        st.markdown(
            f"<div style='background:#1e1e2e;border-radius:8px;padding:20px 24px;margin-top:8px'>"
            f"{row}"
            f"<div style='display:flex;gap:20px;margin-top:14px;flex-wrap:wrap;font-size:0.8rem'>"
            f"<span style='color:#e94560'>Optimizer&nbsp;: Adam</span>"
            f"<span style='color:#a8b2d8'>Loss&nbsp;: sparse_categorical_crossentropy</span>"
            f"<span style='color:#6d7a9f'>Split&nbsp;: 80&nbsp;/&nbsp;20</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── Tab 2 : Architecture MLOps ────────────────────────────────────────────
    with tab_ops:
        st.markdown("#### Flux de données — du déclenchement à la prédiction")

        # ── Ligne 1 : déclenchement ──────────────────────────────────────────
        c1, arr1, c2, arr2, c3 = st.columns([3, 0.6, 2.5, 0.6, 4])
        with c1:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #017cee;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="color:#017cee;font-weight:700;font-size:14px">✈ Airflow DAG</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    Déclenche le training<br>Poll status toutes les 60 s
                </div>
            </div>""", unsafe_allow_html=True)
        with arr1:
            st.markdown("<div style='text-align:center;font-size:13px;color:#a8b2d8;margin-top:18px'>→</div>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #e94560;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="color:#e94560;font-weight:700;font-size:14px">🔑 Gate-API</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    JWT auth<br><code style="font-size:11px">POST /login</code>
                </div>
            </div>""", unsafe_allow_html=True)
        with arr2:
            st.markdown("<div style='text-align:center;font-size:13px;color:#a8b2d8;margin-top:18px'>→</div>",
                        unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #28a745;border-radius:8px;
                        padding:14px 16px">
                <div style="color:#28a745;font-weight:700;font-size:14px">⚙ Train-API</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px;line-height:1.7">
                    <code style="font-size:11px">POST /train</code> → job_id (202)<br>
                    data_loader · preprocess_text · preprocess_image<br>
                    pca_reducer · trainer · save_artifacts
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='text-align:center;font-size:11px;color:#a8b2d8;margin:8px 0'>↓</div>",
                    unsafe_allow_html=True)

        # ── Ligne 2 : sorties du training ────────────────────────────────────
        st.markdown("<div style='color:#a8b2d8;font-size:12px;text-align:center;"
                    "margin-bottom:6px'>Sorties après training</div>",
                    unsafe_allow_html=True)
        o1, o2, o3 = st.columns(3)
        with o1:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #f7981c;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="font-size:1.5rem">📊</div>
                <div style="color:#f7981c;font-weight:700;font-size:14px">DagsHub MLflow</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    Params · métriques<br>datasets · PCAs<br>Model Registry
                </div>
            </div>""", unsafe_allow_html=True)
        with o2:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #28a745;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="font-size:1.5rem">💾</div>
                <div style="color:#28a745;font-weight:700;font-size:14px">Artefacts disque</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    neural_network_model.keras<br>pca_image.pkl · pca_text.pkl<br>text_vectorizer.pkl
                </div>
            </div>""", unsafe_allow_html=True)
        with o3:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #e6522c;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="font-size:1.5rem">📈</div>
                <div style="color:#e6522c;font-weight:700;font-size:14px">Prometheus</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    loss · accuracy<br>drift confidence/entropy<br>Grafana dashboards
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='text-align:center;font-size:11px;color:#a8b2d8;margin:8px 0'>↓</div>",
                    unsafe_allow_html=True)

        # ── Ligne 3 : inférence ──────────────────────────────────────────────
        ui_col, arr3, api_col = st.columns([2.5, 0.6, 4])
        with ui_col:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #ff4b4b;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="font-size:1.5rem">🖥</div>
                <div style="color:#ff4b4b;font-weight:700;font-size:14px">Streamlit UI</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    Saisie texte / image<br>Résultat + probabilités
                </div>
            </div>""", unsafe_allow_html=True)
        with arr3:
            st.markdown("<div style='text-align:center;font-size:13px;color:#a8b2d8;margin-top:20px'>→</div>",
                        unsafe_allow_html=True)
        with api_col:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #a8b2d8;border-radius:8px;
                        padding:14px 16px">
                <div style="color:#a8b2d8;font-weight:700;font-size:14px">🔮 Predict-API</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:8px;
                            display:flex;gap:12px;flex-wrap:wrap">
                    <span><code style="font-size:11px">POST /predict-text</code></span>
                    <span><code style="font-size:11px">POST /predict-image</code></span>
                    <span><code style="font-size:11px">POST /predict-multimodal</code></span>
                    <span><code style="font-size:11px">POST /reload-artifacts</code></span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Gestion du modèle")
        ca, cb, cc = st.columns(3)
        with ca:
            st.markdown("""
            <div style="background:#1e1e2e;border-radius:8px;padding:16px;text-align:center">
                <div style="font-size:2rem">📊</div>
                <b style="color:#e94560">MLflow</b>
                <p style="color:#a8b2d8;font-size:13px;margin:6px 0 0">
                Tracking des runs,<br>params, métriques,<br>datasets, PCAs
                </p>
            </div>
            """, unsafe_allow_html=True)
        with cb:
            st.markdown("""
            <div style="background:#1e1e2e;border-radius:8px;padding:16px;text-align:center">
                <div style="font-size:2rem">🗄️</div>
                <b style="color:#a8b2d8">DagsHub</b>
                <p style="color:#a8b2d8;font-size:13px;margin:6px 0 0">
                Registry distant,<br>artefacts S3,<br>DVC data versioning
                </p>
            </div>
            """, unsafe_allow_html=True)
        with cc:
            st.markdown("""
            <div style="background:#1e1e2e;border-radius:8px;padding:16px;text-align:center">
                <div style="font-size:2rem">🔄</div>
                <b style="color:#a8b2d8">DVC</b>
                <p style="color:#a8b2d8;font-size:13px;margin:6px 0 0">
                Versioning des CSVs<br>et images,<br>pipeline reproductible
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 3 : Stack technique ───────────────────────────────────────────────
    with tab_stack:
        rows = [
            ("Catégorie", "Outil", "Rôle"),
            ("Orchestration", "Apache Airflow 2.x", "DAG de training, polling async, push métriques"),
            ("Feature extraction", "SpaCy + ResNet50", "NLP lemmatisé + CNN pré-entraîné"),
            ("Réduction dim.", "IncrementalPCA (sklearn)", "Batch-safe pour grands datasets"),
            ("Modèle", "Keras / TensorFlow 2.17", "Dense 512 → Dropout → Softmax"),
            ("Experiment tracking", "MLflow 2.17 + DagsHub", "Params, métriques, datasets, modèles"),
            ("Serving", "FastAPI + uvicorn", "predict-api, train-api, gate-api"),
            ("Auth", "JWT (PyJWT)", "Tokens Bearer, RBAC admin/user"),
            ("Monitoring", "Prometheus + Grafana", "Drift confidence/entropy, latence, UP/DOWN"),
            ("Alerting", "Alertmanager", "Slack, email — drift, latence, erreurs"),
            ("Storage", "MinIO + PostgreSQL", "S3 artefacts + meta Airflow/MLflow"),
            ("Data versioning", "DVC + DagsHub S3", "CSVs et images versionnés"),
            ("Containerisation", "Docker Compose", "12 services, réseau bridge dédié"),
        ]
        header = rows[0]
        st.markdown(
            f"<div style=’display:grid;grid-template-columns:1fr 1.5fr 3fr;gap:1px;background:#333;"
            f"border-radius:8px;overflow:hidden’>",
            unsafe_allow_html=True,
        )
        # header
        for cell in header:
            st.markdown(
                f"<div style=’background:#e94560;color:white;padding:8px 12px;font-weight:700;font-size:13px’>{cell}</div>",
                unsafe_allow_html=True,
            )
        for row in rows[1:]:
            bg = "#1e1e2e"
            for i, cell in enumerate(row):
                color = "#e94560" if i == 0 else "#a8b2d8"
                st.markdown(
                    f"<div style=’background:{bg};color:{color};padding:8px 12px;font-size:13px’>{cell}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 4 : Statut des services ────────────────────────────────────────────
    with tab_services:
        st.markdown("#### Statut des services (live)")
        if st.button("Rafraîchir"):
            st.rerun()

        gate_url = os.environ.get("GATE_API_URL", "http://gate-api:5000/login").replace("/login", "")
        predict_url = os.environ.get("PREDICT_API_URL", "http://predict-api:5003")
        train_url = "http://train-api:5002"

        services = [
            ("Streamlit UI",  "http://localhost:8501",  "/"),
            ("Gate-API",      gate_url,                 "/health"),
            ("Predict-API",   predict_url,              "/health"),
            ("Train-API",     train_url,                "/health"),
            ("Prometheus",    "http://prometheus:9090", "/-/healthy"),
            ("Grafana",       "http://grafana:3000",    "/api/health"),
            ("Pushgateway",   "http://pushgateway:9091","/"),
            ("MinIO",         "http://minio:9000",      "/minio/health/live"),
        ]

        cols = st.columns(4)
        for idx, (name, url, path) in enumerate(services):
            try:
                r = requests.get(f"{url}{path}", timeout=3)
                ok = r.status_code in (200, 204)
            except Exception:
                ok = False
            status_txt = "UP" if ok else "DOWN"
            color = "#28a745" if ok else "#dc3545"
            with cols[idx % 4]:
                st.markdown(
                    f"""<div style="background:#1e1e2e;border-radius:8px;padding:14px;
                                    border-top:3px solid {color};text-align:center;margin-bottom:10px">
                        <div style="font-size:1.5rem">{"✅" if ok else "❌"}</div>
                        <div style="color:white;font-weight:600;font-size:14px">{name}</div>
                        <div style="color:{color};font-size:12px;font-weight:700">{status_txt}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.caption("Les services internes (train-api, prometheus…) sont resolus via le réseau Docker.")


def show_prediction_page():
    st.title("MLOps Streamlit Prediction Interface")

    # ── Login form — shown only when not authenticated ────────────────────────
    if "user_token" not in st.session_state:
        st.subheader("Connexion")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            auth_resp = authenticate_user(username, password)
            if auth_resp:
                st.session_state["user_token"] = auth_resp["token"]
                st.session_state["logged_username"] = username
                st.rerun()          # masque immédiatement le formulaire
            else:
                st.error("Identifiants invalides")
        return                      # ne rien afficher d'autre avant connexion

    # ── Authenticated — sidebar info + logout ────────────────────────────────
    uname = st.session_state.get("logged_username", "admin")
    st.sidebar.success(f"Connecté : **{uname}**")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.pop("user_token", None)
        st.session_state.pop("logged_username", None)
        st.rerun()

    # --- Model selection ---
    st.sidebar.subheader("Modèle MLflow")
    available = list_model_versions_and_stages(MODEL_NAME)
    # Numeric versions first, sorted descending (latest first)
    available_sorted = sorted(
        available,
        key=lambda x: (not str(x).isdigit(), -int(x) if str(x).isdigit() else x),
    )

    # Default: latest numeric version, fallback to first item
    numeric_versions = [v for v in available_sorted if str(v).isdigit()]
    default_val = str(max(numeric_versions, key=int)) if numeric_versions else available_sorted[0]
    default_index = available_sorted.index(default_val) if default_val in available_sorted else 0
    stage_or_version = st.sidebar.selectbox("Version / Stage", available_sorted, index=default_index)

    model_uri, version_num, current_stage = get_model_uri(str(stage_or_version))
    if not model_uri:
        st.stop()

    st.sidebar.markdown(f" **Modèle :** `{MODEL_NAME}`")
    st.sidebar.markdown(f" **URI :** `{model_uri}`")
    st.sidebar.markdown(f" **Version :** `{version_num}`  |  **Stage :** `{current_stage}`")

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
                    "image_base64": base64.b64encode(file.getvalue()).decode("utf-8")
                })
            output_file = "./batch_predictions_streamed.jsonl"
            predict_batch_stream(batch_items, st.session_state["user_token"], output_file, model_uri)
        else:
            st.warning("⚠️ Please upload at least one image")



def show_docker_workflow():
    st.title("Infrastructure & Monitoring")

    st.markdown("### Services Docker Compose")
    services_info = [
        ("postgres",      "PostgreSQL 13",         "Backend Airflow + MLflow",    "#6c757d"),
        ("minio",         "MinIO",                  "Stockage S3 artefacts + DVC", "#f7981c"),
        ("gate-api",      "FastAPI + JWT",          "Authentification RBAC",       "#e94560"),
        ("train-api",     "FastAPI + TF 2.17",      "Pipeline training async",     "#e94560"),
        ("predict-api",   "FastAPI + TF 2.17",      "Inférence texte/image/multi", "#e94560"),
        ("airflow",       "Airflow 2.x",            "Orchestration DAG training",  "#017cee"),
        ("prometheus",    "Prometheus 3.x",         "Scraping métriques (15s)",    "#e6522c"),
        ("grafana",       "Grafana",                "Dashboards + drift detection","#f46800"),
        ("alertmanager",  "Alertmanager",           "Routing Slack/email",         "#e6522c"),
        ("pushgateway",   "Pushgateway",            "Métriques batch Airflow",     "#e6522c"),
        ("streamlit",     "Streamlit",              "Interface utilisateur",       "#ff4b4b"),
    ]

    cols = st.columns(3)
    for i, (name, image, role, color) in enumerate(services_info):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#1e1e2e;border-radius:8px;padding:14px;
                        border-left:4px solid {color};margin-bottom:10px">
                <div style="color:{color};font-weight:700;font-size:14px">{name}</div>
                <div style="color:#a8b2d8;font-size:12px">{image}</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:4px">{role}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Métriques de drift exposées à Prometheus")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **predict-api** expose :
        - `prediction_confidence` — confiance max (softmax)
        - `prediction_entropy` — entropie de Shannon
        - `prediction_class_total` — distribution des classes
        - `feature_text_input_mean` / `feature_image_input_mean`
        - `predict_request_latency_seconds`
        """)
    with col2:
        st.markdown("""
        **train-api** expose :
        - `train_loss` / `val_loss` / `train_accuracy` / `val_accuracy`
        - `model_final_val_accuracy` — dernière accuracy finale
        - `training_dataset_size` — taille dataset
        - `model_num_classes` / `epochs_completed_total`
        """)

    st.markdown("### Alertes configurées")
    alerts = [
        ("PredictionConfidenceDrift", "warning", "P50 confiance < 0.40 pendant 15 min"),
        ("PredictionEntropyHigh",     "warning", "P90 entropie > 2.5 nats pendant 15 min"),
        ("ClassDistributionSkewed",   "warning", "Une classe > 80% des prédictions"),
        ("HighPredictionErrorRate",   "critical","Taux d'erreurs 5xx > 10%"),
        ("PredictionLatencyHigh",     "warning", "P95 latence > 5 s"),
        ("ModelValAccuracyLow",       "warning", "val_accuracy < 0.70"),
        ("PredictAPIDown",            "critical","predict-api inaccessible > 3 min"),
        ("TrainAPIDown",              "critical","train-api inaccessible > 3 min"),
    ]
    for name, severity, desc in alerts:
        color = "#dc3545" if severity == "critical" else "#ffc107"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0'>"
            f"<span style='color:{color};font-size:11px;font-weight:700;min-width:70px'>{severity.upper()}</span>"
            f"<code style='color:#a8b2d8;font-size:13px'>{name}</code>"
            f"<span style='color:#9aabb8;font-size:13px'>— {desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ======================
# MAIN
# ======================

# Global font scaling — doubles all rem-based Streamlit text
st.markdown("""
<style>
html { font-size: 100% !important; }
code, .stCode { font-size: 0.85rem !important; }
[data-testid="stSidebarContent"] { font-size: 70% !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Menu")
page = st.sidebar.radio("Navigation", ["Présentation du projet", "Prédictions", "Infrastructure & Monitoring"])

if page == "Présentation du projet":
    show_presentation_page()
elif page == "Prédictions":
    show_prediction_page()
elif page == "Infrastructure & Monitoring":
    show_docker_workflow()
