# streamlit/app_streamlit.py
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
GATE_API_URL    = os.environ.get("GATE_API_URL",    "http://gate-api:5000/login")
PREDICT_API_URL = os.environ.get("PREDICT_API_URL", "http://predict-api:5003")

# --- Test-set ground-truth lookup ---
_TEST_DATA_PATH  = os.environ.get("TEST_DATA_PATH", "/app/data/test_data_zz.xlsx")
_TEST_DATA_LOCAL = r"C:\Users\zobir\DScientest\ds_rakuren\test_data_zz.xlsx"

@st.cache_data(show_spinner=False)
def _load_test_lookup() -> dict:
    for path in (_TEST_DATA_PATH, _TEST_DATA_LOCAL):
        try:
            df = pd.read_excel(path, engine="openpyxl",
                               usecols=["imageid", "designation", "prdtypecode", "category"])
            df["imageid"] = df["imageid"].astype(str).str.strip()
            return df.set_index("imageid").to_dict(orient="index")
        except Exception:
            continue
    return {}


def _lookup_expected(filename: str) -> dict | None:
    lookup = _load_test_lookup()
    if not lookup:
        return None
    for n in re.findall(r"\d{6,}", filename):
        if n in lookup:
            return lookup[n]
    return None


# --- MLflow setup ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "rakuten_model")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# ======================
# UTILS
# ======================
@st.cache_data(ttl=300, show_spinner=False)
def list_model_versions(model_name: str) -> list:
    """Return numeric versions for a registered model, newest first."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return sorted({v.version for v in versions}, key=lambda x: int(x), reverse=True)
    except Exception:
        return []


def authenticate_user(username, password):
    try:
        r = requests.post(GATE_API_URL, json={"username": username, "password": password})
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to authentication service.")
        return None


def predict_single(description, uploaded_image, token, encoder="cv"):
    payload = {}
    if description:
        payload["description"] = description
    if uploaded_image:
        payload["image_base64"] = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")

    if encoder == "ensemble":
        endpoint = f"{PREDICT_API_URL}/predict-ensemble"
        # ensemble always uses multimodal endpoint — model field not needed
    elif uploaded_image and description:
        payload["model"] = encoder
        endpoint = f"{PREDICT_API_URL}/predict-multimodal"
    elif uploaded_image:
        endpoint = f"{PREDICT_API_URL}/predict-image"
    else:
        payload["model"] = encoder
        endpoint = f"{PREDICT_API_URL}/predict-text"

    headers  = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    expected = _lookup_expected(uploaded_image.name) if uploaded_image else None

    try:
        with st.spinner("Running prediction..."):
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=90)
        if resp.status_code == 200:
            result        = resp.json()
            predicted_cat = result.get("category", result["label"])
            predicted_code = result["label"]
            confidence    = f"{100 * max(result['probs'][0]):.0f}%"

            if expected:
                exp_cat  = expected.get("category", "?")
                exp_code = str(expected.get("prdtypecode", "?"))
                match    = predicted_code == exp_code

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"<div style='background:#1e1e2e;border:2px solid #28a745;"
                        f"border-radius:8px;padding:14px;text-align:center'>"
                        f"<div style='color:#9aabb8;font-size:0.75rem'>MODEL PREDICTION</div>"
                        f"<div style='color:#28a745;font-weight:700;font-size:1rem;margin-top:4px'>{predicted_cat}</div>"
                        f"<div style='color:#6d7a9f;font-size:0.75rem'>code {predicted_code}</div>"
                        f"</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(
                        f"<div style='background:#1e1e2e;border:2px solid #a8b2d8;"
                        f"border-radius:8px;padding:14px;text-align:center'>"
                        f"<div style='color:#9aabb8;font-size:0.75rem'>EXPECTED (from description)</div>"
                        f"<div style='color:#a8b2d8;font-weight:700;font-size:1rem;margin-top:4px'>{exp_cat}</div>"
                        f"<div style='color:#6d7a9f;font-size:0.75rem'>code {exp_code}</div>"
                        f"</div>", unsafe_allow_html=True)

                if match:
                    st.caption(f"Match — prediction aligns with description-based expectation. Confidence: {confidence}")
                else:
                    st.caption(
                        f"Mismatch — model predicted **{predicted_cat}**, "
                        f"description suggests **{exp_cat}**. Confidence: {confidence}"
                    )
            else:
                st.success(f"**{predicted_cat}**  —  code `{predicted_code}`  (confidence {confidence})")

            if result.get("encoder") == "ensemble" and "breakdown" in result:
                with st.expander("Per-model breakdown"):
                    bd = result["breakdown"]
                    for enc, info in bd.items():
                        match_main = info["label"] == predicted_code
                        icon = "✅" if match_main else "❌"
                        st.markdown(
                            f"{icon} **{enc.upper()}**: {info['category']} "
                            f"(conf: {info['confidence']:.0%}, weight: {info['weight']:.0%})"
                        )

            if "probs" in result:
                with st.expander("Probabilities per class"):
                    st.write(result["probs"])
            if uploaded_image:
                st.image(uploaded_image, caption=uploaded_image.name, use_column_width=True)
        else:
            st.error(f"API Error: {resp.text}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


def predict_batch_stream(batch_items, token, output_file_path, chunk_size=50,
                         uploaded_files=None):
    """
    Stream batch predictions and display results as an image+table layout.
    uploaded_files: list of Streamlit UploadedFile objects matching batch_items order.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    total   = len(batch_items)
    results = []   # collect parsed results for table display
    progress = st.progress(0, text=f"Predicting 0 / {total}...")

    with output_file_path.open("w", encoding="utf-8") as f_out:
        for i in range(0, total, chunk_size):
            chunk   = batch_items[i:i + chunk_size]
            payload = {"items": chunk, "batch_size": chunk_size}
            try:
                with requests.post(f"{PREDICT_API_URL}/predict-multimodal-batch-stream",
                                   json=payload, headers=headers, stream=True,
                                   timeout=600) as resp:
                    if resp.status_code == 200:
                        for line in resp.iter_lines():
                            if line:
                                decoded = line.decode("utf-8")
                                f_out.write(decoded + "\n")
                                f_out.flush()
                                try:
                                    results.append(json.loads(decoded))
                                except Exception:
                                    pass
                                pct = min(len(results) / total, 1.0)
                                progress.progress(pct,
                                    text=f"Predicting {len(results)} / {total}...")
                    else:
                        st.error(f"API error on chunk {i // chunk_size + 1}: {resp.text}")
            except Exception as e:
                st.error(f"Chunk {i // chunk_size + 1} failed: {e}")

    progress.progress(1.0, text=f"Done — {len(results)} predictions")

    if not results:
        st.warning("No results returned.")
        return

    # ── Results table with thumbnails ────────────────────────────────────────
    st.markdown(f"### Results — {len(results)} predictions")

    # Header row
    h_img, h_file, h_desc, h_cat, h_code, h_conf = st.columns([1, 2, 3, 3, 1, 1])
    h_img.markdown("**Image**")
    h_file.markdown("**Filename**")
    h_desc.markdown("**Description**")
    h_cat.markdown("**Category**")
    h_code.markdown("**Code**")
    h_conf.markdown("**Confidence**")
    st.divider()

    for idx, r in enumerate(results):
        c_img, c_file, c_desc, c_cat, c_code, c_conf = st.columns([1, 2, 3, 3, 1, 1])

        # Thumbnail
        if uploaded_files and idx < len(uploaded_files):
            with c_img:
                st.image(uploaded_files[idx], width=64)
        else:
            c_img.write("—")

        # File name
        fname = uploaded_files[idx].name if uploaded_files and idx < len(uploaded_files) else f"item {idx+1}"
        c_file.markdown(f"<small>{fname}</small>", unsafe_allow_html=True)

        # Description
        desc = batch_items[idx].get("description", "") if idx < len(batch_items) else ""
        c_desc.markdown(f"<small>{desc[:80]}{'…' if len(desc) > 80 else ''}</small>",
                        unsafe_allow_html=True)

        # Category
        category = r.get("category", r.get("label", "?"))
        c_cat.markdown(f"**{category}**")

        # Code
        c_code.markdown(f"`{r.get('label', '?')}`")

        # Confidence
        probs = r.get("probs", [[]])
        if probs and probs[0]:
            conf = max(probs[0])
            color = "#28a745" if conf > 0.6 else "#ffc107" if conf > 0.35 else "#dc3545"
            c_conf.markdown(
                f"<span style='color:{color};font-weight:700'>{conf:.0%}</span>",
                unsafe_allow_html=True,
            )
        else:
            c_conf.write("—")

    st.success(f"Results saved to `{output_file_path.name}`")
    # Download button for the JSONL file
    with open(output_file_path, "rb") as f:
        st.download_button("⬇ Download full JSONL",
                           data=f, file_name=output_file_path.name,
                           mime="application/json")


# ======================
# PAGES
# ======================

def show_presentation_page():
    # Hero
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
                border-radius:12px;padding:36px 32px 28px;margin-bottom:28px">
        <h1 style="color:#e94560;margin:0 0 6px;font-size:2.2rem">
            Rakuten Product Classifier
        </h1>
        <p style="color:#a8b2d8;margin:0;font-size:1.05rem">
            Multimodal MLOps pipeline &mdash; text &amp; image &mdash; from CSV to production model
        </p>
        <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap">
            <span style="background:#e94560;color:white;padding:3px 10px;border-radius:20px;font-size:12px">Python 3.11</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">TensorFlow 2.17</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">MLflow + DagsHub</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">Airflow 2.x</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">Prometheus + Grafana</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">Docker Compose</span>
            <span style="background:#0f3460;color:#a8b2d8;border:1px solid #e94560;padding:3px 10px;border-radius:20px;font-size:12px">jemalloc</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    _cv_acc_str = _clip_acc_str = _ml_acc_str = _mpnet_acc_str = "—"
    _cv_epoch_str = _clip_epoch_str = _ml_epoch_str = _mpnet_epoch_str = "—"
    for _hp, _enc in [
        (Path("/app/data/artifacts/train_history_countvectorizer.json"),        "cv"),
        (Path("/app/data/artifacts/train_history_clip.json"),   "clip"),
        (Path("/app/data/artifacts/train_history_minilm.json"), "minilm"),
        (Path("/app/data/artifacts/train_history_mpnet.json"),  "mpnet"),
    ]:
        if _hp.exists():
            try:
                _hh  = json.loads(_hp.read_text())
                _vas = _hh.get("val_accuracy", [])
                if _vas:
                    _best_val = max(_vas)
                    _best     = f"{_best_val:.1%}"
                    _ep       = str(_vas.index(_best_val) + 1)
                    if _enc == "cv":
                        _cv_acc_str,    _cv_epoch_str    = _best, _ep
                    elif _enc == "clip":
                        _clip_acc_str,  _clip_epoch_str  = _best, _ep
                    elif _enc == "minilm":
                        _ml_acc_str,    _ml_epoch_str    = _best, _ep
                    else:
                        _mpnet_acc_str, _mpnet_epoch_str = _best, _ep
            except Exception:
                pass

    # Row 1 — static dataset stats
    r1c1, r1c2 = st.columns(2)
    r1c1.metric("Product classes", "27",  help="Rakuten categories to predict")
    r1c2.metric("Image features",  "256", help="ResNet50(2048) → PCA(256)")

    # Row 2 — one column per encoder: best val accuracy + epochs as delta
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("TF-IDF + OCR",  _cv_acc_str,    delta=f"{_cv_epoch_str} epochs",
                delta_color="off", help="TF-IDF + OCR + PCA — 84 916 samples, macro F1 early stopping")
    r2c2.metric("CLIP ViT-B/32", _clip_acc_str,  delta=f"{_clip_epoch_str} epochs",
                delta_color="off", help="CLIP ViT-B/32 512-dim — 84 916 samples, macro F1 early stopping")
    r2c3.metric("MiniLM",        _ml_acc_str,    delta=f"{_ml_epoch_str} epochs",
                delta_color="off", help="MiniLM 384-dim — 84 916 samples, macro F1 early stopping")
    r2c4.metric("mpnet",         _mpnet_acc_str, delta=f"{_mpnet_epoch_str} epochs",
                delta_color="off", help="mpnet 768-dim — 84 916 samples, macro F1 early stopping")
    st.divider()

    tab_ml, tab_ops, tab_stack, tab_services = st.tabs([
        "ML Pipeline", "MLOps Architecture", "Tech Stack", "Service Status"
    ])

    # ── Tab 1 : ML Pipeline ───────────────────────────────────────────────────
    with tab_ml:
        st.markdown("#### Feature extraction")
        col_txt, col_img = st.columns(2)

        with col_txt:
            st.markdown("""
            <div style="background:#1e1e2e;border-left:4px solid #e94560;
                        border-radius:8px;padding:18px 20px">
                <b style="color:#e94560">Text branch — 4 encoders</b><br><br>
                <code style="color:#a8b2d8">designation + description + OCR</code><br><br>
                <span style="color:#e94560;font-size:12px;font-weight:600">Encoder A — TF-IDF + OCR</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;SpaCy lemmatisation + Tesseract OCR + TfidfVectorizer</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;IncrementalPCA</span> <code>n_components=512</code><br>
                <b style="color:#28a745">&nbsp;&nbsp;output: 1 × 512</b><br><br>
                <span style="color:#7c3aed;font-size:12px;font-weight:600">Encoder B — CLIP ViT-B/32</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;openai/clip-vit-base-patch32 · L2-normalised</span><br>
                <b style="color:#28a745">&nbsp;&nbsp;output: 1 × 512</b><br><br>
                <span style="color:#a8b2d8;font-size:12px;font-weight:600">Encoder C — MiniLM-L12-v2</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;paraphrase-multilingual-MiniLM-L12-v2</span><br>
                <b style="color:#28a745">&nbsp;&nbsp;output: 1 × 384</b><br><br>
                <span style="color:#f59e0b;font-size:12px;font-weight:600">Encoder D — mpnet-base-v2</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;paraphrase-multilingual-mpnet-base-v2</span><br>
                <b style="color:#28a745">&nbsp;&nbsp;output: 1 × 768</b>
            </div>
            """, unsafe_allow_html=True)

        with col_img:
            st.markdown("""
            <div style="background:#1e1e2e;border-left:4px solid #0f3460;
                        border-radius:8px;padding:18px 20px">
                <b style="color:#a8b2d8">Image branch</b><br><br>
                <code style="color:#a8b2d8">image_jpg</code> <span style="color:#6d7a9f">(224 x 224 px)</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;&nbsp;ResNet50 pre-trained (ImageNet, no top)</span><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;&nbsp;Global Average Pooling →</span> <code>2048 dims</code><br>
                <span style="color:#6d7a9f">&nbsp;&nbsp;&nbsp;IncrementalPCA</span> <code>n_components=256</code><br>
                <b style="color:#28a745">output: 1 × 256</b><br><br>
                <span style="color:#6d7a9f;font-size:11px">Single-pass predict() over full dataset<br>
                (batch_size=64, one generator — eliminates ~2 600 redundant predict calls)</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Classification model — late fusion, 4 variants")
        st.markdown(
            "<div style='background:#1e1e2e;border-radius:8px;padding:20px 24px;margin-top:8px'>"
            "<div style='display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start'>"

            # Text branch
            "<div style='flex:1;min-width:180px;background:#16213e;border:1px solid #e94560;border-radius:6px;padding:10px 12px;font-size:0.78rem'>"
            "<span style='color:#e94560;font-weight:700'>Text branch</span><br>"
            "<span style='color:#6d7a9f'>text features (512/384/768)</span><br>"
            "&#9654; <span style='color:#28a745'>Dense(HIDDEN_1) · ReLU</span><br>"
            "&#9654; <span style='color:#a8b2d8'>LayerNorm</span><br>"
            "&#9654; <span style='color:#a8b2d8'>Dropout(0.35)</span><br>"
            "&#9654; <span style='color:#28a745'>Dense(27) text logits</span>"
            "</div>"

            "<div style='color:#6d7a9f;font-size:1.4rem;align-self:center'>⊕</div>"

            # Image branch
            "<div style='flex:1;min-width:180px;background:#16213e;border:1px solid #4a6fa5;border-radius:6px;padding:10px 12px;font-size:0.78rem'>"
            "<span style='color:#a8b2d8;font-weight:700'>Image branch</span><br>"
            "<span style='color:#6d7a9f'>image PCA features (256)</span><br>"
            "&#9654; <span style='color:#28a745'>Dense(256) · ReLU</span><br>"
            "&#9654; <span style='color:#a8b2d8'>LayerNorm</span><br>"
            "&#9654; <span style='color:#a8b2d8'>Dropout(0.25)</span><br>"
            "&#9654; <span style='color:#28a745'>Dense(27) image logits</span>"
            "</div>"

            "<div style='color:#6d7a9f;font-size:1.4rem;align-self:center'>→</div>"

            # Fusion + output
            "<div style='flex:1;min-width:180px;background:#16213e;border:1px solid #f59e0b;border-radius:6px;padding:10px 12px;font-size:0.78rem'>"
            "<span style='color:#f59e0b;font-weight:700'>Learned gated fusion</span><br>"
            "<span style='color:#6d7a9f'>α = Dense(1, sigmoid) from branch means</span><br>"
            "&#9654; <span style='color:#28a745'>α·softmax(text) + (1-α)·softmax(image)</span><br>"
            "&#9654; <span style='color:#e94560;font-weight:700'>class 1 / 27</span>"
            "</div>"

            "</div>"
            "<div style='display:flex;gap:16px;margin-top:12px;flex-wrap:wrap;font-size:0.8rem'>"
            "<span style='color:#e94560'>Adam lr=0.0002</span>"
            "<span style='color:#a8b2d8'>Focal loss γ∈{1.5,2.0,2.5} per encoder</span>"
            "<span style='color:#6d7a9f'>Stratified 80/20 · Class weights balanced</span>"
            "<span style='color:#6d7a9f'>Early stop patience=8 on val_macro_F1 · ReduceLROnPlateau</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # Training curves
        st.markdown("#### Training curves")
        _artifacts_base = Path("/app/data/artifacts")
        _hist_configs = [
            ("TF-IDF + OCR + PCA",              _artifacts_base / "train_history_countvectorizer.json",        "#e94560"),
            ("CLIP ViT-B/32",                   _artifacts_base / "train_history_clip.json",   "#7c3aed"),
            ("MiniLM (paraphrase-multilingual)", _artifacts_base / "train_history_minilm.json", "#a8b2d8"),
            ("mpnet (paraphrase-multilingual)",  _artifacts_base / "train_history_mpnet.json",  "#f59e0b"),
        ]
        _chart_cols = st.columns(4)
        _any_chart  = False
        for _col, (_enc_label, _hist_file, _color) in zip(_chart_cols, _hist_configs):
            with _col:
                if _hist_file.exists():
                    try:
                        _hdata = json.loads(_hist_file.read_text())
                        _n     = len(_hdata.get("val_accuracy", []))
                        if _n > 0:
                            _any_chart = True
                            _df = pd.DataFrame({
                                "val_accuracy":   _hdata.get("val_accuracy", [None] * _n),
                                "train_accuracy": _hdata.get("accuracy",     [None] * _n),
                            }, index=range(1, _n + 1))
                            _df.index.name = "epoch"
                            _best_val = max(v for v in _hdata["val_accuracy"] if v is not None)
                            st.markdown(
                                f"<div style='background:#1e1e2e;border-left:3px solid {_color};"
                                f"border-radius:6px;padding:8px 14px;margin-bottom:8px'>"
                                f"<span style='color:{_color};font-weight:600'>{_enc_label}</span>"
                                f"<span style='color:#6d7a9f;font-size:12px'> — {_n} epochs "
                                f"| best val acc: <b style=\"color:#28a745\">{_best_val:.1%}</b></span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.line_chart(_df[["val_accuracy", "train_accuracy"]], height=200)
                        else:
                            st.caption(f"{_enc_label}: empty history")
                    except Exception as _e:
                        st.caption(f"{_enc_label}: read error ({_e})")
                else:
                    st.markdown(
                        f"<div style='background:#1e1e2e;border-left:3px solid #333;"
                        f"border-radius:6px;padding:8px 14px;color:#6d7a9f;font-size:13px'>"
                        f"{_enc_label}<br><i>No history — trigger a training run first.</i>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        if not _any_chart:
            st.info("No training history available. Trigger the Airflow DAG to train both models.")

    # ── Tab 2 : MLOps Architecture ────────────────────────────────────────────
    with tab_ops:
        st.markdown("#### Data flow — from trigger to prediction")

        c1, arr1, c2, arr2, c3 = st.columns([3, 0.6, 2.5, 0.6, 4])
        with c1:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #017cee;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="color:#017cee;font-weight:700;font-size:14px">Airflow DAG v7</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    Triggers training · quality gate<br>Cleanlab audit · drift reference<br>Polls every 60 s · last-3-runs log
                </div>
            </div>""", unsafe_allow_html=True)
        with arr1:
            st.markdown("<div style='text-align:center;font-size:13px;color:#a8b2d8;margin-top:18px'>→</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #e94560;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="color:#e94560;font-weight:700;font-size:14px">Gate-API</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    JWT auth<br><code style="font-size:11px">POST /login</code>
                </div>
            </div>""", unsafe_allow_html=True)
        with arr2:
            st.markdown("<div style='text-align:center;font-size:13px;color:#a8b2d8;margin-top:18px'>→</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #28a745;border-radius:8px;
                        padding:14px 16px">
                <div style="color:#28a745;font-weight:700;font-size:14px">Train-API :5002</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px;line-height:1.7">
                    <code style="font-size:11px">POST /train</code> → job_id (202) | 409 if busy<br>
                    Subprocess isolated (jemalloc LD_PRELOAD)<br>
                    PCA cached · data_loader scandir · SpaCy single-pass
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='text-align:center;font-size:11px;color:#a8b2d8;margin:8px 0'>↓</div>", unsafe_allow_html=True)

        st.markdown("<div style='color:#a8b2d8;font-size:12px;text-align:center;margin-bottom:6px'>Training outputs</div>", unsafe_allow_html=True)
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #f7981c;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="color:#f7981c;font-weight:700;font-size:14px">DagsHub MLflow</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    Params · metrics · macro F1 · top-3<br>datasets · PCAs<br>Registry: CV · CLIP · MiniLM · mpnet
                </div>
            </div>""", unsafe_allow_html=True)
        with o2:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #28a745;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="color:#28a745;font-weight:700;font-size:14px">Disk artifacts</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    model.keras · model_clip.keras<br>model_minilm.keras · model_mpnet.keras<br>pca_image/text · tfidf_vectorizer<br>cleanlab_report · drift_reference
                </div>
            </div>""", unsafe_allow_html=True)
        with o3:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #e6522c;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="color:#e6522c;font-weight:700;font-size:14px">Prometheus</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    loss · accuracy<br>drift confidence/entropy<br>per-encoder gauges
                </div>
            </div>""", unsafe_allow_html=True)
        with o4:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #a8b2d8;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="color:#a8b2d8;font-weight:700;font-size:14px">Run log</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    Last 3 runs summary<br>dag_runs.log<br>always written (success + fail)
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='text-align:center;font-size:11px;color:#a8b2d8;margin:8px 0'>↓</div>", unsafe_allow_html=True)

        ui_col, arr3, api_col = st.columns([2.5, 0.6, 4])
        with ui_col:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #ff4b4b;border-radius:8px;
                        padding:14px 16px;text-align:center">
                <div style="color:#ff4b4b;font-weight:700;font-size:14px">Streamlit UI</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:6px">
                    Text / image input<br>Result + probabilities
                </div>
            </div>""", unsafe_allow_html=True)
        with arr3:
            st.markdown("<div style='text-align:center;font-size:13px;color:#a8b2d8;margin-top:20px'>→</div>", unsafe_allow_html=True)
        with api_col:
            st.markdown("""
            <div style="background:#1e1e2e;border:1px solid #a8b2d8;border-radius:8px;
                        padding:14px 16px">
                <div style="color:#a8b2d8;font-weight:700;font-size:14px">Predict-API :5003</div>
                <div style="color:#9aabb8;font-size:12px;margin-top:8px;display:flex;gap:12px;flex-wrap:wrap">
                    <span><code style="font-size:11px">POST /predict-text?model=cv|clip|minilm|mpnet</code></span>
                    <span><code style="font-size:11px">POST /predict-image</code></span>
                    <span><code style="font-size:11px">POST /predict-multimodal</code> (modality fallback)</span>
                    <span><code style="font-size:11px">POST /gradcam</code></span>
                    <span><code style="font-size:11px">POST /reload-artifacts</code></span>
                </div>
                <div style="color:#6d7a9f;font-size:11px;margin-top:6px">
                    4 models loaded · modality fallback · GradCAM · Evidently drift &nbsp;|&nbsp; jemalloc LD_PRELOAD
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Model management")
        ca, cb, cc = st.columns(3)
        with ca:
            st.markdown("""
            <div style="background:#1e1e2e;border-radius:8px;padding:16px;text-align:center">
                <b style="color:#e94560">MLflow</b>
                <p style="color:#a8b2d8;font-size:13px;margin:6px 0 0">
                Run tracking, params,<br>metrics, datasets, PCAs.<br>
                Models: CV · CLIP<br>MiniLM · mpnet
                </p>
            </div>""", unsafe_allow_html=True)
        with cb:
            st.markdown("""
            <div style="background:#1e1e2e;border-radius:8px;padding:16px;text-align:center">
                <b style="color:#a8b2d8">DagsHub</b>
                <p style="color:#a8b2d8;font-size:13px;margin:6px 0 0">
                Remote registry,<br>S3 artifacts,<br>DVC data versioning
                </p>
            </div>""", unsafe_allow_html=True)
        with cc:
            st.markdown("""
            <div style="background:#1e1e2e;border-radius:8px;padding:16px;text-align:center">
                <b style="color:#a8b2d8">DVC</b>
                <p style="color:#a8b2d8;font-size:13px;margin:6px 0 0">
                CSV + image versioning,<br>reproducible pipeline,<br>dual-stage dvc.yaml
                </p>
            </div>""", unsafe_allow_html=True)

    # ── Tab 3 : Tech Stack ────────────────────────────────────────────────────
    with tab_stack:
        rows = [
            ("Category",           "Tool",                                   "Role"),
            ("Orchestration",      "Apache Airflow 2.x (DAG v7)",            "Training · quality gate · cleanlab · drift reference · last-3-runs log"),
            ("Text encoding A",    "SpaCy + TF-IDF + Tesseract OCR",         "Lemmatised NLP + image OCR text, TfidfVectorizer(sublinear_tf), PCA 512-d"),
            ("Text encoding B",    "CLIP ViT-B/32 (laion/clip)",             "Vision-language embeddings 512-d, dedicated clip-encoder service"),
            ("Text encoding C",    "MiniLM-L12-v2 (sentence-transformers)",  "Multilingual sentence embeddings 384-d, dedicated minilm-encoder service"),
            ("Text encoding D",    "mpnet-base-v2 (sentence-transformers)",  "Multilingual sentence embeddings 768-d, shared minilm-encoder service"),
            ("Image encoding",     "ResNet50 (Keras)",                       "ImageNet pretrained, global avg pool → 2048-d, PCA 256-d"),
            ("Fusion",             "Late fusion (learned gated average)",     "Separate text/image branches → per-class logits → α·text + (1-α)·image"),
            ("Loss",               "Focal loss (Lin et al. 2017)",            "γ per encoder (CV:2.5, CLIP:1.5, MiniLM/mpnet:2.0); focal + class weights"),
            ("Dim. reduction",     "IncrementalPCA (sklearn)",                "Memory-safe batched PCA — fingerprint cache, skipped on re-runs"),
            ("Experiment tracking","MLflow 2.17 + DagsHub",                  "Params, val_acc, macro_F1, top-3 per epoch — 4 registered models"),
            ("Label quality",      "Cleanlab 2.6.5",                         "Confident learning audit post-training — flags mislabeled samples"),
            ("Quality gate",       "pytest (post-training)",                  "Accuracy floors, macro F1 floor, no class collapse, overfit gap ≤ 0.20"),
            ("Drift monitoring",   "Evidently AI 0.4.33",                    "PSI-based feature drift — bounded 2000-row buffer, 10 HTML reports max"),
            ("Explainability",     "GradCAM (predict-api)",                  "Class-discriminative heatmaps via analytical weight projection through PCA"),
            ("Serving",            "FastAPI + uvicorn",                      "predict-api (4 models + GradCAM + fallback), train-api (409 guard), gate-api"),
            ("Auth",               "JWT (PyJWT)",                            "Bearer tokens, RBAC admin/user"),
            ("Memory allocator",   "jemalloc (LD_PRELOAD)",                  "Replaces glibc — eliminates TF/NumPy heap corruption during training"),
            ("Monitoring",         "Prometheus + Grafana",                   "Drift confidence/entropy, per-encoder accuracy, latency, UP/DOWN"),
            ("Alerting",           "Alertmanager",                           "Email (Brevo SMTP) — drift, latency, accuracy (CV≥0.72, CLIP≥0.80), down"),
            ("Storage",            "MinIO + PostgreSQL",                     "S3 artifacts + Airflow/MLflow metadata"),
            ("Data versioning",    "DVC + DagsHub S3",                      "Feature caches + models versioned, ocr_text.csv tracked"),
            ("Containerisation",   "Docker Compose",                         "14 services, static subnet 172.20.0.0/16, jobs volume-mounted"),
        ]
        header = rows[0]
        st.markdown(
            "<div style='display:grid;grid-template-columns:1.2fr 2fr 3.5fr;gap:1px;background:#333;"
            "border-radius:8px;overflow:hidden'>",
            unsafe_allow_html=True,
        )
        for cell in header:
            st.markdown(
                f"<div style='background:#e94560;color:white;padding:8px 12px;font-weight:700;font-size:13px'>{cell}</div>",
                unsafe_allow_html=True,
            )
        for row in rows[1:]:
            for i, cell in enumerate(row):
                color = "#e94560" if i == 0 else "#a8b2d8"
                st.markdown(
                    f"<div style='background:#1e1e2e;color:{color};padding:8px 12px;font-size:13px'>{cell}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 4 : Service Status ─────────────────────────────────────────────────
    with tab_services:
        st.markdown("#### Service status (live)")
        if st.button("Refresh"):
            st.rerun()

        gate_url    = os.environ.get("GATE_API_URL",    "http://gate-api:5000/login").replace("/login", "")
        predict_url = os.environ.get("PREDICT_API_URL", "http://predict-api:5003")

        services = [
            ("Streamlit UI",    "http://localhost:8501",      "/"),
            ("Gate-API",        gate_url,                     "/health"),
            ("Predict-API",     predict_url,                  "/health"),
            ("Train-API",       "http://train-api:5002",      "/health"),
            ("MiniLM Encoder",  "http://minilm-encoder:5004", "/health"),
            ("CLIP Encoder",    "http://clip-encoder:5007",   "/health"),
            ("Prometheus",      "http://prometheus:9090",     "/-/healthy"),
            ("Grafana",         "http://grafana:3000",        "/api/health"),
            ("Pushgateway",     "http://pushgateway:9091",    "/"),
            ("MinIO",           "http://minio:9000",          "/minio/health/live"),
        ]

        cols = st.columns(3)
        for idx, (name, url, path) in enumerate(services):
            try:
                r  = requests.get(f"{url}{path}", timeout=3)
                ok = r.status_code in (200, 204)
            except Exception:
                ok = False
            status_txt = "UP" if ok else "DOWN"
            color      = "#28a745" if ok else "#dc3545"
            with cols[idx % 3]:
                st.markdown(
                    f"<div style='background:#1e1e2e;border-radius:8px;padding:14px;"
                    f"border-top:3px solid {color};text-align:center;margin-bottom:10px'>"
                    f"<div style='color:white;font-weight:600;font-size:14px'>{name}</div>"
                    f"<div style='color:{color};font-size:12px;font-weight:700'>{status_txt}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.caption("Internal services (train-api, prometheus, etc.) are resolved via the Docker network.")


def show_prediction_page():
    st.title("Prediction Interface")

    if "user_token" not in st.session_state:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Log in"):
            auth_resp = authenticate_user(username, password)
            if auth_resp:
                st.session_state["user_token"]     = auth_resp["token"]
                st.session_state["logged_username"] = username
                st.rerun()
            else:
                st.error("Invalid credentials")
        return

    uname = st.session_state.get("logged_username", "admin")
    st.sidebar.success(f"Logged in: **{uname}**")
    if st.sidebar.button("Log out"):
        st.session_state.pop("user_token",     None)
        st.session_state.pop("logged_username", None)
        st.rerun()

    st.sidebar.subheader("MLflow Models")
    for _mname in ["rakuten_multimodal_cv", "rakuten_multimodal_clip",
                   "rakuten_multimodal_minilm", "rakuten_multimodal_mpnet"]:
        _vers = list_model_versions(_mname)
        _latest = _vers[0] if _vers else "—"
        st.sidebar.caption(f"{_mname}: v{_latest}")

    # Read best val_accuracy for each encoder from training history
    _enc_history_paths = {
        "cv":     Path("/app/data/artifacts/train_history_countvectorizer.json"),
        "clip":   Path("/app/data/artifacts/train_history_clip.json"),
        "minilm": Path("/app/data/artifacts/train_history_minilm.json"),
        "mpnet":  Path("/app/data/artifacts/train_history_mpnet.json"),
    }
    _enc_best_acc = {}
    for _enc, _hp in _enc_history_paths.items():
        try:
            _vas = json.loads(_hp.read_text()).get("val_accuracy", [])
            if _vas:
                _enc_best_acc[_enc] = max(_vas)
        except Exception:
            pass

    def _acc_tag(enc):
        return f" · {_enc_best_acc[enc]:.1%}" if enc in _enc_best_acc else ""

    _encoder_options = {
        "Ensemble (all 4 models — weighted average)": "ensemble",
        f"TF-IDF + OCR + PCA (512-d){_acc_tag('cv')}":     "cv",
        f"CLIP ViT-B/32 (512-d){_acc_tag('clip')}":         "clip",
        f"MiniLM multilingual (384-d){_acc_tag('minilm')}": "minilm",
        f"mpnet multilingual (768-d){_acc_tag('mpnet')}":   "mpnet",
    }

    # Default to ensemble
    _default_idx = 0

    st.sidebar.subheader("Text encoder")
    encoder_label    = st.sidebar.radio(
        "Text model", list(_encoder_options.keys()), index=_default_idx
    )
    selected_encoder = _encoder_options[encoder_label]

    _enc_info = {
        "ensemble": "Weighted average of all 4 models. Weights proportional to val accuracy. Most robust — reduces single-model failures.",
        "cv":       "TF-IDF + OCR + spaCy lemmatisation → IncrementalPCA (512-d).",
        "clip":     "openai/clip-vit-base-patch32 — 512 dims, English-focused, best visual features.",
        "minilm":   "paraphrase-multilingual-MiniLM-L12-v2 — 384 dims, multilingual.",
        "mpnet":    "paraphrase-multilingual-mpnet-base-v2 — 768 dims, multilingual.",
    }
    st.sidebar.info(_enc_info[selected_encoder])

    st.subheader("Single Prediction")
    description    = st.text_input("Product description")
    uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if st.button("Predict"):
        predict_single(description, uploaded_image, st.session_state["user_token"],
                       encoder=selected_encoder)

    st.subheader("Batch Prediction (Streaming)")
    uploaded_files    = st.file_uploader("Upload multiple images", accept_multiple_files=True,
                                         type=["png", "jpg", "jpeg"])
    descriptions_file = st.file_uploader("Optional: CSV with descriptions", type=["csv"])
    if st.button("Run Batch"):
        if uploaded_files:
            batch_items  = []
            descriptions = []
            if descriptions_file:
                df_desc = pd.read_csv(descriptions_file, encoding="utf-8-sig",
                                      encoding_errors="replace")
                if "description" in df_desc.columns:
                    descriptions = df_desc["description"].tolist()
            for idx, file in enumerate(uploaded_files):
                batch_items.append({
                    "description":  descriptions[idx] if idx < len(descriptions) else "",
                    "image_base64": base64.b64encode(file.getvalue()).decode("utf-8"),
                })
            predict_batch_stream(batch_items, st.session_state["user_token"],
                                 "./batch_predictions_streamed.jsonl",
                                 uploaded_files=uploaded_files)
        else:
            st.warning("Please upload at least one image.")


def show_drift_reports():
    """Drift monitoring tab — shows Evidently HTML reports and buffer status."""
    st.markdown("### Evidently AI — Input Drift Reports")

    _drift_dir = Path("/app/data/artifacts/drift_reports")
    _ref_path  = Path("/app/data/artifacts/drift_reference.csv")

    # ── Status row ────────────────────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    reports = sorted(_drift_dir.glob("drift_*.html"), key=lambda p: p.stat().st_mtime) \
              if _drift_dir.exists() else []
    ref_ok  = _ref_path.exists() and _ref_path.stat().st_size > 1000
    s1.metric("Reports on disk",    len(reports),          help="Max 10 kept, oldest auto-deleted")
    s2.metric("Reference built",    "Yes" if ref_ok else "No",
              help="5k stratified sample from training data")
    s3.metric("Buffer capacity",    "2 000 predictions",
              help="Report auto-generated when buffer fills; trigger manually below")

    st.divider()

    # ── Manual trigger ────────────────────────────────────────────────────────
    if "user_token" in st.session_state:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            if st.button("⚡ Trigger Report Now", help="Generates report from current prediction buffer"):
                try:
                    resp = requests.post(
                        f"{PREDICT_API_URL}/drift-trigger-report",
                        headers={"Authorization": f"Bearer {st.session_state['user_token']}"},
                        timeout=10,
                    )
                    if resp.ok:
                        st.success("Report generation triggered — refresh in ~30 s")
                    else:
                        st.warning(f"Trigger returned {resp.status_code}")
                except Exception as e:
                    st.error(f"Could not reach predict-api: {e}")
        with col_info:
            st.caption(
                "Reports are auto-generated when 2 000 multimodal predictions accumulate. "
                "The button triggers generation immediately regardless of buffer size."
            )
    else:
        st.info("Log in on the Predictions page to trigger reports manually.")

    st.divider()

    # ── Report list + inline viewer ───────────────────────────────────────────
    if not reports:
        st.info(
            "No drift reports yet. Make multimodal predictions via the Prediction page "
            "or click **Trigger Report Now** above after logging in."
        )
        return

    st.markdown(f"**{len(reports)} report(s) available** (newest first):")

    for rpt in reversed(reports):
        ts_str = rpt.stem.replace("drift_", "")
        try:
            from datetime import datetime
            ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            ts = ts_str
        size_kb = round(rpt.stat().st_size / 1024)

        with st.expander(f"📊  {ts}  ({size_kb} KB)", expanded=(rpt == reports[-1])):
            col_dl, col_view = st.columns([1, 3])
            with col_dl:
                with open(rpt, "rb") as f:
                    st.download_button(
                        label="⬇ Download HTML",
                        data=f,
                        file_name=rpt.name,
                        mime="text/html",
                        key=f"dl_{rpt.name}",
                    )
            with col_view:
                st.caption(f"Path on host: `data/artifacts/drift_reports/{rpt.name}`")

            # Render report inline as an iframe
            html_content = rpt.read_text(encoding="utf-8", errors="replace")
            st.components.v1.html(html_content, height=600, scrolling=True)


def show_docker_workflow():
    st.title("Infrastructure & Monitoring")

    st.markdown("### Docker Compose Services")
    services_info = [
        ("postgres",       "PostgreSQL 13",                   "Airflow + MLflow backend",                "#6c757d"),
        ("minio",          "MinIO",                           "S3 artifact storage + DVC remote",        "#f7981c"),
        ("gate-api",       "FastAPI + JWT",                   "RBAC authentication",                     "#e94560"),
        ("train-api",      "FastAPI + TF 2.17 + jemalloc",   "Async training pipeline (subprocess)",    "#e94560"),
        ("minilm-encoder", "FastAPI + sentence-transformers", "Bulk MiniLM text encoding to .npy cache",   "#a8b2d8"),
        ("clip-encoder",   "FastAPI + transformers",          "Bulk CLIP ViT-B/32 text encoding to .npy",  "#7c3aed"),
        ("predict-api",    "FastAPI + TF 2.17 + jemalloc",   "4-model inference + GradCAM + modality fallback + Evidently drift", "#e94560"),
        ("airflow",        "Airflow 2.x",                    "DAG v7 — training, quality gate, cleanlab, drift reference",  "#017cee"),
        ("prometheus",     "Prometheus 3.x",                  "Metrics scraping (15 s), 8 targets",      "#e6522c"),
        ("grafana",        "Grafana",                         "Dashboards + per-encoder drift detection","#f46800"),
        ("alertmanager",   "Alertmanager",                    "Email routing via Brevo SMTP",            "#e6522c"),
        ("pushgateway",    "Pushgateway",                     "Batch metrics from Airflow",              "#e6522c"),
        ("streamlit",      "Streamlit",                       "User interface",                          "#ff4b4b"),
        ("mlflow",         "MLflow (DagsHub external)",       "Experiment tracking + Model Registry",    "#f7981c"),
    ]
    cols = st.columns(3)
    for i, (name, image, role, color) in enumerate(services_info):
        with cols[i % 3]:
            st.markdown(
                f"<div style='background:#1e1e2e;border-radius:8px;padding:14px;"
                f"border-left:4px solid {color};margin-bottom:10px'>"
                f"<div style='color:{color};font-weight:700;font-size:14px'>{name}</div>"
                f"<div style='color:#a8b2d8;font-size:12px'>{image}</div>"
                f"<div style='color:#9aabb8;font-size:12px;margin-top:4px'>{role}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Drift metrics exposed to Prometheus")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **predict-api** exposes:
        - `prediction_confidence` — max softmax probability
        - `prediction_entropy` — Shannon entropy
        - `prediction_class_total` — class distribution
        - `feature_text_input_mean` / `feature_image_input_mean`
        - `predict_request_latency_seconds`
        """)
    with col2:
        st.markdown("""
        **train-api** exposes (labeled by encoder):
        - `train_loss` / `val_loss` / `train_accuracy` / `val_accuracy`
        - `model_final_val_accuracy{encoder="cv|clip|minilm|mpnet"}` — final val accuracy
        - `model_final_val_loss{encoder="cv|clip|minilm|mpnet"}` — final val loss
        - `training_dataset_size` — dataset size
        - `model_num_classes` / `epochs_completed_total`
        """)

    st.markdown("### Configured alerts")
    alerts = [
        ("CVModelValAccuracyLow",     "warning",  "CV (TF-IDF+OCR) val_accuracy < 0.72"),
        ("CLIPModelValAccuracyLow",   "warning",  "CLIP val_accuracy < 0.70"),
        ("MiniLMModelValAccuracyLow", "warning",  "MiniLM val_accuracy < 0.70"),
        ("mpnetModelValAccuracyLow",  "warning",  "mpnet val_accuracy < 0.72"),
        ("PredictionConfidenceDrift", "warning",  "P50 confidence < 0.40 for 15 min"),
        ("PredictionEntropyHigh",     "warning",  "P90 entropy > 2.5 nats for 15 min"),
        ("ClassDistributionSkewed",   "warning",  "One class > 80% of predictions"),
        ("HighPredictionErrorRate",   "critical", "5xx error rate > 10%"),
        ("PredictionLatencyHigh",     "warning",  "P95 latency > 5 s"),
        ("PredictAPIDown",            "critical", "predict-api unreachable > 3 min"),
        ("TrainAPIDown",              "critical", "train-api unreachable > 3 min"),
        ("MiniLMEncoderDown",         "warning",  "minilm-encoder unreachable > 3 min"),
        ("CLIPEncoderDown",           "warning",  "clip-encoder unreachable > 3 min"),
        ("GateAPIDown",               "critical", "gate-api (auth) unreachable > 2 min"),
        ("DiskSpaceLow",              "critical", "Root filesystem < 10% free"),
        ("HighMemoryUsage",           "warning",  "System memory > 90% for 10 min"),
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
st.markdown("""
<style>
html { font-size: 100% !important; }
code, .stCode { font-size: 0.85rem !important; }
[data-testid="stSidebarContent"] { font-size: 70% !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "Project Overview",
    "Predictions",
    "Infrastructure & Monitoring",
    "Drift Reports",
])

if page == "Project Overview":
    show_presentation_page()
elif page == "Predictions":
    show_prediction_page()
elif page == "Infrastructure & Monitoring":
    show_docker_workflow()
elif page == "Drift Reports":
    show_drift_reports()
