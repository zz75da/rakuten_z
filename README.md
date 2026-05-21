# Rakuten MLOps Platform

[![CI with DVC + Tests](https://github.com/zz75da/rakuten_z/actions/workflows/dvc-ci.yml/badge.svg?branch=main)](https://github.com/zz75da/rakuten_z/actions/workflows/dvc-ci.yml)

End-to-end MLOps platform for **multimodal product classification** (text + image → 27 categories).  
Built with FastAPI microservices, Apache Airflow, MLflow / DagsHub, and a full Prometheus / Grafana monitoring stack.

---

## Table of Contents

- [Model Overview](#model-overview)
- [Architecture](#architecture)
- [Services](#services)
- [Quick Start](#quick-start)
- [Service Endpoints](#service-endpoints)
- [Monitoring & Drift Detection](#monitoring--drift-detection)
- [Data & Experiment Tracking](#data--experiment-tracking)
- [Test Suite](#test-suite)
- [Repository Structure](#repository-structure)
- [Environment Variables](#environment-variables)

---

## Model Overview

Three text encoders are available. The image branch and classifier architecture are shared across all three.

### Encoder A — CountVectorizer (default)

```
Text description ──► SpaCy lemmatise ──► CountVectorizer (10 000) ──► IncrementalPCA (512-d) ─┐
                                                                                                ├─► hstack (896-d) ──► Dense 512 ──► Dropout ──► Dense 256 ──► Dropout ──► Dense 27 (softmax)
Product image    ──► ResNet50 (2 048-d)                             ──► IncrementalPCA (384-d) ─┘
```

Registered in MLflow as **`rakuten_multimodal_cv`**.

### Encoder B — CLIP ViT-B/32

```
Text description ──► openai/clip-vit-base-patch32 (CLIPTextModel) ──► L2-norm (512-d) ─────────┐
                                                                                                ├─► hstack (896-d) ──► Dense 512 ──► Dropout ──► Dense 256 ──► Dropout ──► Dense 27 (softmax)
Product image    ──► ResNet50 (2 048-d) ──► IncrementalPCA (384-d) ──────────────────────────┘
```

Registered in MLflow as **`rakuten_multimodal_clip`**.

### Encoder C — MiniLM (multilingual)

```
Text description ──► paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers) ──► 384-d ─┐
                                                                                                ├─► hstack (768-d) ──► Dense 512 ──► Dropout ──► Dense 256 ──► Dropout ──► Dense 27 (softmax)
Product image    ──► ResNet50 (2 048-d) ──► IncrementalPCA (384-d) ──────────────────────────┘
```

Registered in MLflow as **`rakuten_multimodal_minilm`**.

| Component | CountVectorizer | CLIP ViT-B/32 | MiniLM |
|-----------|-----------------|---------------|--------|
| Text encoding | CountVectorizer (10 000) → PCA (512) | CLIPTextModel → L2-norm (512-d) | MiniLM-L12-v2 → 384-d |
| Combined input dim | 896 | 896 | 768 |
| Classifier | Dense 512 → Dropout 0.45 → Dense 256 → Dropout 0.35 → Softmax 27 | same | same |
| Image encoder | ResNet50 (ImageNet, global avg pool, no top) → PCA (384) | same | same |
| Encoding service | inline (train-api) | clip-encoder :5007 | minilm-encoder :5004 |
| Training stability | jemalloc `LD_PRELOAD` — prevents TF/glibc allocator conflicts | same | same |

---

## Architecture

```
┌─────────────┐     JWT      ┌─────────────┐
│  Streamlit  │◄────────────►│  gate-api   │  Authentication & token validation
│    :8501    │              │    :5000    │
└──────┬──────┘              └─────────────┘
       │ Bearer token
       ▼
┌─────────────┐  POST /train   ┌────────────────────────────────────────────────┐
│  Airflow    │───────────────►│  train-api  :5002                              │
│    :8080    │◄─ poll status  │  ├── run_full_pipeline.py (subprocess)         │
│  DAG v6     │                │  │   ├── preprocess_text / preprocess_image    │
└─────────────┘                │  │   ├── pca_reducer (subprocess)              │
       │                       │  │   └── trainer (TF, jemalloc)                │
       ├── POST /encode ──────►│  clip-encoder  :5007 (CLIP only)              │
       │                       │  minilm-encoder :5004 (MiniLM only)           │
       │                       └───────────────┬────────────────────────────────┘
       │                                       │ POST /reload-artifacts
       │                                       ▼
       │                            ┌──────────────────┐     ┌──────────────┐
       │                            │  predict-api     │────►│  MLflow      │
       │                            │    :5003         │     │  / DagsHub   │
       │                            │  model_cv        │     └──────────────┘
       │                            │  model_clip      │
       │                            │  model_minilm    │
       │                            └──────────────────┘
       │                            /predict-text  ?model=cv|clip|minilm
       │                            /predict-image
       │                            /predict-multimodal

┌──────────────────────────────────────────────────────┐
│  Prometheus (:9090)  ◄──  scrapes all FastAPI apps   │
│  Grafana    (:3000)  ──►  drift dashboard            │
│  Alertmanager(:9093) ──►  confidence / accuracy / UP alerts │
└──────────────────────────────────────────────────────┘
```

---

## Services

| Service | External port | Role |
|---------|--------------|------|
| **gate-api** | 5000 | JWT authentication (`/login`, `/validate-token`) |
| **train-api** | 5002 | Async model training — CV, CLIP, and MiniLM sequential runs via Airflow DAG |
| **predict-api** | 5003 | Multimodal inference — serves CV, CLIP, and MiniLM models simultaneously |
| **minilm-encoder** | 5005 | Bulk MiniLM sentence encoding (`POST /encode`, `GET /status`) |
| **clip-encoder** | 5006 | Bulk CLIP ViT-B/32 text encoding (`POST /encode`, `GET /status`, `DELETE /cache`) |
| **airflow** | 8080 | DAG v6 orchestration: CV → CLIP encode → CLIP train → MiniLM encode → MiniLM train |
| **streamlit** | 8501 | Interactive demo UI + pipeline presentation + live training curves |
| **mlflow** | — | Experiment tracking hosted on DagsHub (external) |
| **minio** | 9002 | S3-compatible object storage for MLflow artifacts |
| **prometheus** | 9090 | Metrics collection |
| **grafana** | 3000 | Dashboards & alerting |
| **alertmanager** | 9093 | Alert routing (email) |

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- ~20 GB RAM and ~20 GB free disk for the full stack (all three encoders)

### 1 — Clone and configure

```bash
git clone https://github.com/zz75da/rakuten_z.git
cd rakuten_z
cp .env.template .env          # fill in DAGSHUB_USER, DAGSHUB_TOKEN, etc.
```

### 2 — Pull DVC-tracked data

```bash
pip install dvc dvc-s3
dvc pull                      # downloads datasets and model artifacts from DagsHub
```

### 3 — Start the stack

```bash
docker-compose up -d --build
```

### 4 — Verify services are healthy

```bash
docker-compose ps
curl http://localhost:5000/health   # gate-api
curl http://localhost:5002/health   # train-api
curl http://localhost:5003/health   # predict-api
curl http://localhost:5005/health   # minilm-encoder
curl http://localhost:5006/health   # clip-encoder
```

### 5 — Trigger a training run (via Airflow)

Open `http://localhost:8080`, enable the `rakuten_multimodal_pipeline_v6` DAG and trigger it manually.  
The DAG runs CV → CLIP → MiniLM sequentially (~6 h on first run; ~4 h on subsequent runs with warm caches).

---

## Service Endpoints

### gate-api — Authentication

```
POST /login              {"username": "admin", "password": "admin_pass"}
POST /validate-token     Authorization: Bearer <token>
GET  /health
GET  /metrics
```

### train-api — Async Training

```
POST /train              {
                           "use_dev_images": false,
                           "epochs": 60,
                           "batch_size": 128,
                           "text_encoder": "countvectorizer"  // or "clip" | "minilm"
                         }
                         → 202 {"job_id": "...", "status": "running"}
                         → 409 if another job is already running

GET  /train/status/{id}  → {"status": "success|running|failed|interrupted",
                             "step": "text_features|pca|training|...",
                             "final_metrics": {...}, "mlflow_run_id": "..."}
GET  /health
GET  /metrics
```

### clip-encoder — Bulk CLIP Text Encoding

```
POST   /encode           Encodes data/X_train_update.csv → text_features_clip.npy
                         (idempotent — validates cache params before skipping)
GET    /status           → {"status": "idle|encoding|done|error", "message": "..."}
DELETE /cache            Invalidate cache — forces re-encode on next POST /encode
GET    /health           → includes cache_valid, normalize_embeddings, batch_size
```

### minilm-encoder — Bulk MiniLM Encoding

```
POST /encode             Encodes data/X_train_update.csv → text_features_minilm.npy
GET  /status             → {"status": "idle|encoding|done|error", "message": "..."}
GET  /health
```

### predict-api — Inference

```
POST /predict-text        {"description": "leather handbag", "model": "cv"}
                          model: "cv" (default) | "clip" | "minilm"
POST /predict-image       {"image_base64": "<base64 JPEG>"}
POST /predict-multimodal  {"description": "...", "image_base64": "...", "model": "cv"}
POST /reload-artifacts    reload all models + PCA + vectorizer from disk
GET  /health              → includes cv_model_loaded, clip_model_loaded, minilm_model_loaded
GET  /metrics
```

Each prediction endpoint returns:
```json
{
  "pred_class": 40,
  "label": "40",
  "category": "Movies & DVDs",
  "probs": [[0.02, 0.85, ...]],
  "mode": "text_only | image_only | multimodal",
  "encoder": "cv | clip | minilm"
}
```

All three models (`model_cv`, `model_clip`, `model_minilm`) are loaded simultaneously at startup if available.  
predict-api uses `LD_PRELOAD=jemalloc` so TF and sentence-transformers/transformers can coexist.

---

## Monitoring & Drift Detection

Grafana is auto-provisioned with the **Rakuten Drift Dashboard** at `http://localhost:3000`.

### Prometheus metrics collected

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_confidence` | Histogram | Max softmax probability per prediction |
| `prediction_entropy` | Histogram | Shannon entropy (high = uncertain = possible drift) |
| `prediction_class_total` | Counter | Predictions per class label (distribution drift) |
| `feature_text_input_mean` | Gauge | Mean of text feature vector (last prediction) |
| `feature_image_input_mean` | Gauge | Mean of image feature vector (last prediction) |
| `model_final_val_accuracy{encoder="cv\|clip\|minilm"}` | Gauge | Val accuracy after last training run |
| `model_final_val_loss{encoder="cv\|clip\|minilm"}` | Gauge | Val loss after last training run |
| `training_dataset_size` | Gauge | Number of samples used in the last training run |

### Alert rules (`monitoring/alert-rules.yml`)

- `CVModelValAccuracyLow` / `CLIPModelValAccuracyLow` / `MiniLMModelValAccuracyLow` — val accuracy below 0.70
- `CLIPEncoderDown` / `MiniLMEncoderDown` — encoder service unreachable > 3 min
- `PredictionConfidenceDrift` — P50 confidence drops below 0.40 for 15 min
- `PredictionEntropyHigh` — P90 entropy exceeds 2.5 nats
- `ClassDistributionSkewed` — single class exceeds 80% of recent predictions
- `PredictionLatencyHigh` — P95 latency above 5 s
- `DiskSpaceLow` — root filesystem below 10% free

---

## Data & Experiment Tracking

| Resource | URL |
|----------|-----|
| Code repository | https://github.com/zz75da/rakuten_z |
| DagsHub (data + MLflow) | https://dagshub.com/zz75da/rakuten_z |
| MLflow experiments | https://dagshub.com/zz75da/rakuten_z/experiments |
| Model Registry | https://dagshub.com/zz75da/rakuten_z/models |

### MLflow experiments

Each training run logs:

**Parameters** (visible as columns in DagsHub):

| Parameter | Description |
|-----------|-------------|
| `text_encoder` | `countvectorizer`, `clip`, or `minilm` |
| `model_name` | `rakuten_multimodal_cv`, `_clip`, or `_minilm` |
| `input_dim` | Combined feature vector size (896 for CV/CLIP, 768 for MiniLM) |
| `preprocess.pca_components` | Image PCA output dim (currently 384) |
| `preprocess.n_text_pca_components` | CV text PCA dim (512); N/A for CLIP and MiniLM |
| `dataset_rows` | Training set size |
| `epochs_max` | Configured max epochs |
| `model.hidden_1`, `model.hidden_2` | Classifier hidden layer sizes |
| `model.learning_rate`, `model.l2_reg` | Optimiser settings |

**Metrics:** `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy` (per epoch) + `final_val_accuracy`

**Artefacts:**

| File | Encoder |
|------|---------|
| `neural_network_model.keras` | CV |
| `neural_network_model_clip.keras` | CLIP |
| `neural_network_model_minilm.keras` | MiniLM |
| `train_history.json` / `_clip.json` / `_minilm.json` | per encoder |
| `pca_image.pkl` | shared (all encoders) |
| `pca_text.pkl` | CV only |
| `text_vectorizer.pkl` | CV only |
| `label_encoder.pkl` | shared |

### MLflow Model Registry

Three separate registered models:

- **`rakuten_multimodal_cv`** — CountVectorizer + PCA encoder
- **`rakuten_multimodal_clip`** — CLIP ViT-B/32 text encoder
- **`rakuten_multimodal_minilm`** — MiniLM multilingual encoder

Each version is tagged with `encoder`, `task`, `dataset`, and `framework`.  
Run names follow `cv_train_YYYYMMDD_HHMM` / `clip_train_...` / `minilm_train_...`.

---

## Test Suite

```bash
# Run against the airflow container (has all dependencies)
docker exec airflow python -m pytest tests/ -q
```

**Current coverage:** unit tests for all three encoders across artifacts, PCA reducer, model architecture, and training workflow. TF-dependent tests skip cleanly without a real TF runtime.

### Test files

| File | Scope | What it tests |
|------|-------|---------------|
| `tests/unit/test_gate_api.py` | Unit | Login, JWT claims, token validation |
| `tests/unit/test_predict_api.py` | Unit | All predict endpoints, tri-model globals, drift metrics, reload, 503 for unavailable models |
| `tests/unit/test_train_api.py` | Unit | Async /train, RBAC, job registry, status polling, 409 concurrent-training guard |
| `tests/unit/test_artifacts.py` | Unit | `save_artifacts()` for CV, CLIP, and MiniLM — file sets, skip/overwrite, round-trip |
| `tests/unit/test_models.py` | Unit | Keras architecture for all three input dims (896 CV, 896 CLIP, 768 MiniLM) |
| `tests/unit/test_pca_reducer.py` | Unit | `reduce_features()` shape, dtype, encoder-specific filename, MiniLM/CLIP pass-through |
| `tests/unit/test_preprocess_text.py` | Unit | Text cleaning + CountVectorizer properties |
| `tests/unit/test_preprocess_image.py` | Unit | ResNet50 output shape (skipped without TF) |
| `tests/unit/test_preprocess.py` | Unit | Cross-module smoke tests |
| `tests/integration/test_api_integration.py` | Integration | JWT login → validate cross-service flow |
| `tests/integration/test_pipeline.py` | Integration | Text → PCA mini-pipeline, determinism |
| `tests/integration/test_workflow.py` | Integration | Full async training job lifecycle, subprocess architecture, 409 guard |

---

## Repository Structure

```
rakuten_mlops_services/
├── airflow/
│   ├── dags/train_dag_v6.py        # DAG v6: CV → CLIP encode → CLIP train → MiniLM encode → MiniLM train
│   ├── Dockerfile
│   └── requirements.txt
├── gate-api/
│   ├── app.py                      # JWT auth service
│   ├── Dockerfile
│   └── requirements.txt
├── train-api/
│   ├── app.py                      # async training API (409 guard, job persistence)
│   ├── services/
│   │   ├── data_loader.py
│   │   ├── preprocess_text.py
│   │   ├── preprocess_image.py
│   │   ├── pca_reducer.py          # IncrementalPCA — CV PCA, MiniLM/CLIP pass-through
│   │   ├── run_pca.py              # PCA subprocess (writes result to temp JSON)
│   │   ├── run_full_pipeline.py    # full training subprocess (TF isolated from uvicorn)
│   │   ├── trainer.py              # MLflow logging, encoder-specific Keras model + filenames
│   │   └── artifacts.py           # save_artifacts() — CV, CLIP, and MiniLM file sets
│   ├── Dockerfile                  # tensorflow:2.17.0 + libjemalloc2
│   └── requirements.txt
├── clip-encoder/
│   ├── app.py                      # FastAPI service: atomic CLIP encoding with param-aware cache
│   ├── Dockerfile                  # python:3.11-slim + CPU-only torch + transformers
│   └── requirements.txt
├── minilm-encoder/
│   ├── app.py                      # FastAPI service: bulk MiniLM encoding to .npy cache
│   ├── encode.py
│   ├── Dockerfile                  # python:3.11-slim + CPU-only torch
│   └── requirements.txt
├── predict-api/
│   ├── app.py                      # tri-model inference: model_cv + model_clip + model_minilm
│   ├── Dockerfile                  # python:3.11-slim + libjemalloc2
│   └── requirements.txt
├── streamlit/
│   ├── app_streamlit.py            # demo UI + pipeline presentation + live training curves
│   └── requirements.txt
├── monitoring/
│   ├── prometheus.yml              # scrape config (mounted by Prometheus)
│   ├── alert-rules.yml             # CV + CLIP + MiniLM accuracy + encoder UP/DOWN alerts
│   ├── alertmanager.yml.tmpl
│   └── grafana_dashboards/
│       └── rakuten_drift_dashboard.json  # encoder dropdown includes cv|clip|minilm
├── tests/
│   ├── conftest.py
│   ├── unit/                       # 9 unit test modules
│   ├── integration/                # 3 integration test modules
│   └── requirements-test.txt
├── dvc.yaml                        # 3 text-encoding + 3 PCA + 3 training + 1 predict stages
├── docker-compose.yml              # 14 services, static subnet 172.20.0.0/16
├── params.yaml                     # all tunable hyperparameters incl. clip.* section
└── pytest.ini
```

---

## Environment Variables

Copy `.env.template` to `.env` and fill in:

| Variable | Default | Description |
|----------|---------|-------------|
| `DAGSHUB_USER` | — | DagsHub username (DVC remote + MLflow auth) |
| `DAGSHUB_TOKEN` | — | DagsHub access token |
| `MLFLOW_EXPERIMENT_NAME` | `rakuten_z` | MLflow experiment name |
| `MLFLOW_MODEL_NAME` | `rakuten_multimodal` | Base name — suffixed `_cv`, `_clip`, or `_minilm` at registration |
| `ARTIFACTS_PATH` | `/app/data/artifacts` | Path to serialised model artefacts (predict-api) |
| `GATE_API_URL` | `http://gate-api:5000` | Internal gate-api address |
| `PREDICT_API_URL` | `http://predict-api:5003` | Internal predict-api address |
| `LD_PRELOAD` | `/usr/lib/x86_64-linux-gnu/libjemalloc.so.2` | jemalloc — prevents TF/glibc heap corruption |
