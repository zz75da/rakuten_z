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

The classifier combines two feature branches that are reduced independently before being fused and fed to a dense classifier:

```
Text description ──► CountVectorizer (5 000) ──► IncrementalPCA (1 024-d)  ─┐
                                                                              ├─► concat (1 324-d) ──► Dense 512 ──► Dropout ──► Dense 27 (softmax)
Product image    ──► ResNet50 (2 048-d)       ──► IncrementalPCA  (300-d)  ─┘
```

| Component | Detail |
|-----------|--------|
| Text encoder | scikit-learn `CountVectorizer` (max 5 000 features) + French/English stop-word removal via spaCy |
| Image encoder | Keras `ResNet50` pretrained on ImageNet (global average pooling, no top) |
| Dimensionality reduction | `IncrementalPCA` — memory-efficient, batch-fitted |
| Classifier | Keras `Dense(512, relu) → Dropout(0.5) → Dense(27, softmax)` |
| Output | 27 Rakuten product categories |

---

## Architecture

```
┌─────────────┐     JWT      ┌─────────────┐
│  Streamlit  │◄────────────►│  gate-api   │  Authentication & token validation
│     UI      │              │  :5000      │
└──────┬──────┘              └─────────────┘
       │ Bearer token
       ▼
┌─────────────┐    POST /train  ┌─────────────┐    logs     ┌──────────────┐
│  Airflow    │────────────────►│  train-api  │────────────►│  MLflow      │
│  DAG        │◄── poll status  │  :8000      │             │  / DagsHub   │
└─────────────┘                 └──────┬──────┘             └──────────────┘
                                       │ reload artifacts
                                       ▼
                               ┌─────────────┐
                               │ predict-api │  /predict-text
                               │  :8001      │  /predict-image
                               └─────────────┘  /predict-multimodal

┌──────────────────────────────────────────────────────┐
│  Prometheus (:9090)  ◄──  scrapes all FastAPI apps   │
│  Grafana    (:3000)  ──►  22-panel drift dashboard   │
│  Alertmanager(:9093) ──►  confidence / accuracy alerts│
└──────────────────────────────────────────────────────┘
```

---

## Services

| Service | Port | Role |
|---------|------|------|
| **gate-api** | 5000 | JWT authentication (`/login`, `/validate-token`) |
| **train-api** | 8000 | Async model training (`POST /train` → 202 + job_id, `GET /train/status/{id}`) |
| **predict-api** | 8001 | Multimodal inference (`/predict-text`, `/predict-image`, `/predict-multimodal`) |
| **mlflow** | 5001 | Experiment tracking & model registry |
| **airflow** | 8080 | DAG orchestration (trigger → poll → evaluate) |
| **streamlit** | 8501 | Interactive demo UI |
| **prometheus** | 9090 | Metrics collection |
| **grafana** | 3000 | Dashboards & alerting |
| **alertmanager** | 9093 | Alert routing |

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- ~8 GB RAM available for the full stack

### 1 — Clone and configure

```bash
git clone https://github.com/zz75da/rakuten_z.git
cd rakuten_z
cp .env.example .env          # fill in DAGSHUB_USER, DAGSHUB_TOKEN, etc.
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
curl http://localhost:8000/health   # train-api
curl http://localhost:8001/health   # predict-api
```

---

## Service Endpoints

### gate-api — Authentication

```
POST /login              {"username": "admin", "password": "admin_pass"}
POST /validate-token     Authorization: Bearer <token>
GET  /health
GET  /metrics            Prometheus scrape endpoint
```

### train-api — Async Training

```
POST /train              {"use_dev_images": true, "epochs": 10, "batch_size": 32}
                         → 202 {"job_id": "...", "status": "running"}
GET  /train/status/{id}  → {"status": "success|running|failed", "final_metrics": {...}}
GET  /health
GET  /metrics
```

Training runs in a background thread.  Airflow polls `/train/status/{job_id}` every 60 s
with a 4-hour timeout and up to 10 consecutive failure retries.

### predict-api — Inference

```
POST /predict-text        {"description": "leather handbag"}
POST /predict-image       {"image_base64": "<base64 JPEG>"}
POST /predict-multimodal  {"description": "...", "image_base64": "..."}
POST /reload-artifacts    reload model + PCA + vectorizer from disk/MLflow
GET  /health
GET  /metrics
```

Each prediction endpoint returns:
```json
{
  "pred_class": 40,
  "label": "product_40",
  "probs": [[0.02, 0.85, ...]],
  "mode": "text_only | image_only | multimodal"
}
```

---

## Monitoring & Drift Detection

Grafana is auto-provisioned with a **22-panel Rakuten Drift Dashboard** at `http://localhost:3000`.

### Prometheus metrics collected

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_confidence` | Histogram | Max softmax probability per prediction |
| `prediction_entropy` | Histogram | Shannon entropy (high = uncertain = possible drift) |
| `prediction_class_total` | Counter | Predictions per class label (distribution drift) |
| `feature_text_input_mean` | Gauge | Mean of text feature vector (last prediction) |
| `feature_image_input_mean` | Gauge | Mean of image feature vector (last prediction) |
| `model_final_accuracy` | Gauge | Val accuracy of the last completed training run |
| `training_dataset_size` | Gauge | Number of samples used in the last training run |

### Alert rules (`monitoring/alert-rules.yml`)

- `PredictionConfidenceDrift` — mean confidence drops below 0.5 for 5 min
- `PredictionEntropyHigh` — mean entropy exceeds 2.5 (high uncertainty)
- `ClassDistributionSkewed` — any single class exceeds 50 % of predictions
- `ModelValAccuracyLow` — validation accuracy below 0.60
- `PredictionLatencyHigh` — p95 latency above 2 s

---

## Data & Experiment Tracking

| Resource | URL |
|----------|-----|
| Code repository | https://github.com/zz75da/rakuten_z |
| DagsHub (data + MLflow) | https://dagshub.com/zz75da/rakuten_z |

### DVC workflow

```bash
dvc pull                  # download latest data & artifacts
dvc repro                 # re-run the full pipeline if inputs changed
dvc push                  # upload new artifacts after training
```

### MLflow experiments

Training runs log:
- Parameters: `n_components_img`, `pca_text_n_components`, `epochs`, `batch_size`, `copy`
- Metrics: `accuracy`, `val_accuracy`, `loss`, `val_loss` (per epoch)
- Artefacts: `neural_network_model.keras`, `pca_image.pkl`, `pca_text.pkl`, `text_vectorizer.pkl`, `label_encoder.pkl`
- Dataset inputs via `mlflow.log_input`

---

## Test Suite

```bash
# Recommended — Python 3.12 on host
py -3.12 -m pytest tests/unit/        # unit tests only  (~30 s)
py -3.12 -m pytest tests/integration/ # integration tests (~3 min, spacy required)
py -3.12 -m pytest tests/             # full suite
```

**Current status: 96 passed, 9 skipped** (TensorFlow not required on host — image tests skip cleanly).

### Test files

| File | Scope | What it tests |
|------|-------|---------------|
| `tests/unit/test_gate_api.py` | Unit | Login, JWT claims, token validation |
| `tests/unit/test_predict_api.py` | Unit | All predict endpoints, drift metrics, reload |
| `tests/unit/test_train_api.py` | Unit | Async /train, RBAC, job registry, status polling |
| `tests/unit/test_artifacts.py` | Unit | save_artifacts() — create / skip / overwrite |
| `tests/unit/test_models.py` | Unit | Vectorizer, LabelEncoder, Keras architecture |
| `tests/unit/test_pca_reducer.py` | Unit | reduce_features() shape, dtype, PCA objects |
| `tests/unit/test_preprocess_text.py` | Unit | Text cleaning + CountVectorizer properties |
| `tests/unit/test_preprocess_image.py` | Unit | ResNet50 output shape (skipped without TF) |
| `tests/unit/test_preprocess.py` | Unit | Cross-module smoke tests |
| `tests/integration/test_api_integration.py` | Integration | JWT login → validate cross-service flow |
| `tests/integration/test_pipeline.py` | Integration | Text → PCA mini-pipeline, determinism |
| `tests/integration/test_workflow.py` | Integration | Full async training job lifecycle |

Test dependencies are in `tests/requirements-test.txt`.  
Log files with the **last 3 runs** per category are written automatically to `tests/logs/`.

---

## Repository Structure

```
rakuten_mlops_services/
├── airflow/
│   ├── dags/train_dag.py          # poll-based async training DAG
│   ├── Dockerfile
│   └── requirements.txt
├── gate-api/
│   ├── app.py                     # JWT auth service
│   ├── Dockerfile
│   └── requirements.txt
├── train-api/
│   ├── app.py                     # async training API
│   ├── services/
│   │   ├── data_loader.py
│   │   ├── preprocess_text.py
│   │   ├── preprocess_image.py
│   │   ├── pca_reducer.py
│   │   ├── trainer.py             # MLflow logging, Keras model
│   │   └── artifacts.py
│   ├── Dockerfile
│   └── requirements.txt
├── predict-api/
│   ├── app.py                     # multimodal inference + drift metrics
│   ├── Dockerfile
│   └── requirements.txt
├── mlflow_docker/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── requirements.txt
├── streamlit/
│   ├── app_streamlit.py           # demo UI + service status
│   └── requirements.txt
├── monitoring/
│   ├── prometheus.yml
│   ├── alert-rules.yml
│   ├── alertmanager.yml
│   └── grafana_dashboards/
│       └── rakuten_drift_dashboard.json
├── grafana/
│   └── provisioning/datasources/
│       └── prometheus.yml         # auto-provisioned datasource
├── tests/
│   ├── conftest.py                # shared fixtures + rotating log plugin
│   ├── unit/                      # 9 unit test modules
│   ├── integration/               # 3 integration test modules
│   ├── logs/                      # last-3-runs log files (auto-generated)
│   └── requirements-test.txt
├── docker-compose.yml
├── pytest.ini
└── requirements.txt
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `MLFLOW_TRACKING_USERNAME` | — | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | — | DagsHub token |
| `MLFLOW_EXPERIMENT_NAME` | `rakuten_z` | MLflow experiment name |
| `MLFLOW_MODEL_NAME` | `rakuten_multimodal` | Registered model name |
| `DAGSHUB_USER` | — | DagsHub username (DVC remote) |
| `DAGSHUB_TOKEN` | — | DagsHub access token |
| `ARTIFACTS_PATH` | `/app/data/artifacts` | Path to serialised model artefacts |
| `GATE_API_URL` | `http://gate-api:5000` | Internal gate-api address |
