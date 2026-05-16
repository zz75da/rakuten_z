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

Two text encoders are available. The image branch and classifier architecture are shared.

### Encoder A — CountVectorizer (default)

```
Text description ──► SpaCy lemmatise ──► CountVectorizer (5 000) ──► IncrementalPCA (1 024-d) ─┐
                                                                                                  ├─► hstack (1 324-d) ──► Dense 512 ──► Dropout ──► Dense 256 ──► Dropout ──► Dense 27 (softmax)
Product image    ──► ResNet50 (2 048-d)                            ──► IncrementalPCA  (300-d) ─┘
```

Registered in MLflow as **`rakuten_multimodal_cv`**.

### Encoder B — MiniLM (multilingual)

```
Text description ──► paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers) ──► 384-d ─┐
                                                                                                 ├─► hstack (684-d) ──► Dense 512 ──► Dropout ──► Dense 256 ──► Dropout ──► Dense 27 (softmax)
Product image    ──► ResNet50 (2 048-d) ──► IncrementalPCA (300-d) ──────────────────────────┘
```

Registered in MLflow as **`rakuten_multimodal_minilm`**.

| Component | CountVectorizer variant | MiniLM variant |
|-----------|-------------------------|----------------|
| Text encoding | CountVectorizer (5 000) → PCA (1 024) | MiniLM-L12-v2 → 384-d |
| Combined input dim | 1 324 | 684 |
| Classifier | Dense 512 → Dropout 0.3 → Dense 256 → Dropout 0.2 → Softmax 27 | same |
| Image encoder | ResNet50 (ImageNet, global avg pool, no top) → PCA (300) | same |
| Training stability | jemalloc `LD_PRELOAD` — prevents TF/glibc allocator conflicts | same |
| Subprocess isolation | Full pipeline runs in `run_full_pipeline.py` child process — uvicorn heap never touched by TF | same |

---

## Architecture

```
┌─────────────┐     JWT      ┌─────────────┐
│  Streamlit  │◄────────────►│  gate-api   │  Authentication & token validation
│    :8501    │              │    :5004    │
└──────┬──────┘              └─────────────┘
       │ Bearer token
       ▼
┌─────────────┐  POST /train   ┌────────────────────────────────────────────┐
│  Airflow    │───────────────►│  train-api  :5002                          │
│    :8080    │◄─ poll status  │  ├── run_full_pipeline.py (subprocess)     │
└─────────────┘                │  │   ├── preprocess_text / preprocess_image│
                               │  │   ├── pca_reducer (subprocess)          │
                               │  │   └── trainer (TF, jemalloc)            │
                               │  └── minilm-encoder :5005 (MiniLM only)   │
                               └───────────────┬────────────────────────────┘
                                               │ POST /reload-artifacts
                                               ▼
                                    ┌──────────────────┐     ┌──────────────┐
                                    │  predict-api     │────►│  MLflow      │
                                    │    :5003         │     │  / DagsHub   │
                                    │  model_cv  ─┐   │     └──────────────┘
                                    │  model_minilm─┘  │
                                    └──────────────────┘
                                    /predict-text  ?model=cv|minilm
                                    /predict-image
                                    /predict-multimodal

┌──────────────────────────────────────────────────────┐
│  Prometheus (:9090)  ◄──  scrapes all FastAPI apps   │
│  Grafana    (:3000)  ──►  22-panel drift dashboard   │
│  Alertmanager(:9093) ──►  confidence / accuracy alerts│
└──────────────────────────────────────────────────────┘
```

---

## Services

| Service | External port | Role |
|---------|--------------|------|
| **gate-api** | 5004 | JWT authentication (`/login`, `/validate-token`) |
| **train-api** | 5002 | Async model training — CV and MiniLM sequential runs via Airflow DAG |
| **minilm-encoder** | 5005 | Bulk MiniLM sentence encoding (`POST /encode`, `GET /status`) |
| **predict-api** | 5003 | Multimodal inference — serves both CV and MiniLM models simultaneously |
| **airflow** | 8080 | DAG orchestration (encode → train CV → train MiniLM → push artifacts) |
| **streamlit** | 8501 | Interactive demo UI + pipeline presentation + live training curves |
| **mlflow** | — | Experiment tracking hosted on DagsHub (external) |
| **minio** | 9002 | S3-compatible object storage for MLflow artifacts |
| **prometheus** | 9090 | Metrics collection |
| **grafana** | 3000 | Dashboards & alerting |
| **alertmanager** | 9093 | Alert routing (email / Slack) |

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- ~12 GB RAM and ~15 GB free disk for the full stack

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
curl http://localhost:5004/health   # gate-api
curl http://localhost:5002/health   # train-api
curl http://localhost:5003/health   # predict-api
curl http://localhost:5005/health   # minilm-encoder
```

### 5 — Trigger a training run (via Airflow)

Open `http://localhost:8080`, enable the `rakuten_multimodal_pipeline_v5_1` DAG and trigger it manually.  
The DAG runs CV training then MiniLM training sequentially (~3–4 h on a mid-range laptop).

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
                           "epochs": 10,
                           "batch_size": 32,
                           "text_encoder": "countvectorizer"  // or "minilm"
                         }
                         → 202 {"job_id": "...", "status": "running"}
                         → 409 if another job is already running

GET  /train/status/{id}  → {"status": "success|running|failed|interrupted",
                             "step": "text_features|pca|training|...",
                             "final_metrics": {...}, "mlflow_run_id": "..."}
GET  /health
GET  /metrics
```

Training runs in an isolated subprocess (`run_full_pipeline.py`) with `LD_PRELOAD=jemalloc` to prevent TF allocator crashes.  
Airflow polls `/train/status/{job_id}` every 60 s with a 4-hour timeout.  
Only one job runs at a time — a second `POST /train` returns **409** while a job is in progress.

### minilm-encoder — Bulk Sentence Encoding

```
POST /encode             Encodes data/X_train_update.csv → text_features_minilm.npy
                         (idempotent — skips if cache already exists)
GET  /status             → {"status": "idle|encoding|done|error", "message": "..."}
GET  /health
```

### predict-api — Inference

```
POST /predict-text        {"description": "leather handbag", "model": "cv"}
                          model: "cv" (default) | "minilm"
POST /predict-image       {"image_base64": "<base64 JPEG>"}
POST /predict-multimodal  {"description": "...", "image_base64": "...", "model": "cv"}
POST /reload-artifacts    reload all models + PCA + vectorizer from disk
GET  /health
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
  "encoder": "cv | minilm"
}
```

Both `model_cv` and `model_minilm` are loaded simultaneously at startup.  
predict-api uses `LD_PRELOAD=jemalloc` so TF and sentence-transformers can coexist in the same process.

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
| MLflow experiments | https://dagshub.com/zz75da/rakuten_z/experiments |
| Model Registry | https://dagshub.com/zz75da/rakuten_z/models |

### DVC workflow

```bash
dvc pull                  # download latest data & artifacts
dvc repro                 # re-run the full pipeline if inputs changed
dvc push                  # upload new artifacts after training
```

### MLflow experiments

Each training run (CV or MiniLM) logs:

**Parameters** (visible as columns in DagsHub experiments):

| Parameter | Description |
|-----------|-------------|
| `text_encoder` | `countvectorizer` or `minilm` |
| `model_name` | `rakuten_multimodal_cv` or `rakuten_multimodal_minilm` |
| `input_dim` | Combined feature vector size (1324 or 684) |
| `dataset_rows` | Training set size |
| `epochs_max` / `actual_epochs_trained` | Configured vs early-stopped |
| `batch_size`, `learning_rate` | Optimiser settings |
| `pca_image_components`, `pca_text_components` | PCA reduction sizes |

**Metrics:** `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy` (per epoch) + `final_val_accuracy`

**Artefacts:**

| File | Encoder |
|------|---------|
| `neural_network_model.keras` | CV |
| `neural_network_model_minilm.keras` | MiniLM |
| `train_history.json` | CV |
| `train_history_minilm.json` | MiniLM |
| `pca_image.pkl`, `pca_text.pkl` | CV (image PCA shared) |
| `text_vectorizer.pkl`, `label_encoder.pkl` | both |

### MLflow Model Registry

Both models are registered as separate named models:

- **`rakuten_multimodal_cv`** — CountVectorizer + PCA encoder
- **`rakuten_multimodal_minilm`** — MiniLM multilingual encoder

Each version is tagged with `encoder`, `task`, `dataset`, and `framework`.  
Run names follow the pattern `cv_train_YYYYMMDD_HHMM` / `minilm_train_YYYYMMDD_HHMM`.

---

## Test Suite

```bash
# Run inside the predict-api container (correct dependency environment)
docker run --rm -v "$(pwd):/proj" -w /proj \
  rakuten_mlops_services-predict-api:latest \
  sh -c "pip install pytest pyjwt pillow -q && python -m pytest tests/unit/ -q"
```

**Current status: 36 unit tests passing** (TF-dependent tests skip cleanly without a real TF runtime).

### Test files

| File | Scope | What it tests |
|------|-------|---------------|
| `tests/unit/test_gate_api.py` | Unit | Login, JWT claims, token validation |
| `tests/unit/test_predict_api.py` | Unit | All predict endpoints, dual-model globals, drift metrics, reload, encoder field, 503 for unavailable MiniLM |
| `tests/unit/test_train_api.py` | Unit | Async /train, RBAC, job registry, status polling, **409 concurrent-training guard** |
| `tests/unit/test_artifacts.py` | Unit | `save_artifacts()` for CV and MiniLM — file sets, skip/overwrite, round-trip |
| `tests/unit/test_models.py` | Unit | Vectorizer, LabelEncoder, Keras architecture for **both** input dims (1324 / 684) |
| `tests/unit/test_pca_reducer.py` | Unit | `reduce_features()` shape, dtype, PCA objects |
| `tests/unit/test_preprocess_text.py` | Unit | Text cleaning + CountVectorizer properties |
| `tests/unit/test_preprocess_image.py` | Unit | ResNet50 output shape (skipped without TF) |
| `tests/unit/test_preprocess.py` | Unit | Cross-module smoke tests |
| `tests/integration/test_api_integration.py` | Integration | JWT login → validate cross-service flow |
| `tests/integration/test_pipeline.py` | Integration | Text → PCA mini-pipeline, determinism |
| `tests/integration/test_workflow.py` | Integration | Full async training job lifecycle, **409 concurrent rejection** |

Log files with the **last 3 runs** per category are written automatically to `tests/logs/`.

---

## Repository Structure

```
rakuten_mlops_services/
├── airflow/
│   ├── dags/train_dag.py          # poll-based async DAG: encode → train CV → train MiniLM
│   ├── Dockerfile
│   └── requirements.txt
├── gate-api/
│   ├── app.py                     # JWT auth service
│   ├── Dockerfile
│   └── requirements.txt
├── train-api/
│   ├── app.py                     # async training API (409 guard, job persistence)
│   ├── services/
│   │   ├── data_loader.py
│   │   ├── preprocess_text.py
│   │   ├── preprocess_image.py
│   │   ├── pca_reducer.py         # IncrementalPCA, batch-safe
│   │   ├── run_pca.py             # PCA subprocess (writes result to temp JSON file)
│   │   ├── run_full_pipeline.py   # full training subprocess (TF isolated from uvicorn)
│   │   ├── trainer.py             # MLflow logging, Keras model, encoder-specific filenames
│   │   └── artifacts.py          # save_artifacts() — CV and MiniLM file sets
│   ├── Dockerfile                 # tensorflow:2.17.0 + libjemalloc2
│   └── requirements.txt
├── minilm-encoder/
│   ├── app.py                     # FastAPI service: bulk MiniLM encoding to .npy cache
│   ├── encode.py
│   ├── Dockerfile                 # python:3.11-slim + CPU-only torch
│   └── requirements.txt
├── predict-api/
│   ├── app.py                     # dual-model inference: model_cv + model_minilm
│   ├── Dockerfile                 # python:3.11-slim + libjemalloc2
│   └── requirements.txt
├── streamlit/
│   ├── app_streamlit.py           # demo UI + pipeline presentation + live training curves
│   └── requirements.txt
├── monitoring/
│   ├── prometheus.yml
│   ├── alert-rules.yml
│   ├── alertmanager.yml.tmpl
│   └── grafana_dashboards/
│       └── rakuten_drift_dashboard.json
├── grafana/
│   └── provisioning/datasources/
│       └── prometheus.yml
├── tests/
│   ├── conftest.py                # shared fixtures + rotating log plugin
│   ├── unit/                      # 9 unit test modules (36 tests)
│   ├── integration/               # 3 integration test modules
│   ├── logs/                      # last-3-runs log files (auto-generated)
│   └── requirements-test.txt
├── dvc.yaml                       # data pipeline definition (displayed in DagsHub)
├── docker-compose.yml
├── pytest.ini
└── params.yaml
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Default | Description |
|----------|---------|-------------|
| `DAGSHUB_USER` | — | DagsHub username (DVC remote + MLflow auth) |
| `DAGSHUB_TOKEN` | — | DagsHub access token |
| `MLFLOW_EXPERIMENT_NAME` | `rakuten_z` | MLflow experiment name |
| `MLFLOW_MODEL_NAME` | `rakuten_multimodal` | Base name — suffixed `_cv` or `_minilm` at registration |
| `ARTIFACTS_PATH` | `/app/data/artifacts` | Path to serialised model artefacts (predict-api) |
| `GATE_API_URL` | `http://gate-api:5000` | Internal gate-api address |
| `PREDICT_API_URL` | `http://predict-api:5003` | Internal predict-api address (used by train-api after training) |
| `LD_PRELOAD` | `/usr/lib/x86_64-linux-gnu/libjemalloc.so.2` | jemalloc allocator — prevents TF/glibc heap corruption in train-api and predict-api |
