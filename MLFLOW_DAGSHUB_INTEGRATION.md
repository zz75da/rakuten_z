# MLflow + DagsHub Integration Setup Guide

## Overview
This guide documents the integration of MLflow with DagsHub S3 storage for the Rakuten MLOps platform. All services now work seamlessly with centralized experiment tracking and artifact storage.

## Architecture Changes

### 1. MLflow Backend
- **Tracking Server**: PostgreSQL (persistent database for experiments, runs, metrics)
- **Artifact Storage**: DagsHub S3 (compatible with AWS S3 API)
- **Endpoint**: `http://mlflow:5000`

### 2. Service Updates

#### train-api
- Now initializes MLflow tracking on startup
- Logs experiments to DagsHub via MLflow
- Tracks training metrics (loss, accuracy, epochs)
- All artifacts versioned in MLflow

#### predict-api
- Attempts to load models from MLflow registry first
- Falls back to disk artifacts if MLflow unavailable
- Supports loading from `Production` stage models

#### Removed Services
- **preprocess-api**: Preprocessing is now embedded in `train-api`
- All references removed from DAGs, monitoring, and alerts

## Configuration

### Environment Variables (.env)

```dotenv
# DagsHub / S3 Credentials
AWS_ACCESS_KEY_ID=zz75da
AWS_SECRET_ACCESS_KEY=f157b23c5416f005713a6b28e5c46847f4ff838e

# MLflow Configuration
MLFLOW_TRACKING_URI=https://dagshub.com/zz75da/rakuten_z.mlflow
MLFLOW_S3_ENDPOINT_URL=https://dagshub.com/api/v1/repo-buckets/s3/zz75da
MLFLOW_ARTIFACT_URI=s3://rakuten_z
MLFLOW_TRACKING_USERNAME=zz75da
MLFLOW_TRACKING_PASSWORD=f157b23c5416f005713a6b28e5c46847f4ff838e

# Experiment Tracking
MLFLOW_EXPERIMENT_NAME=rakuten_z

# DagsHub
DAGSHUB_USER=zz75da
DAGSHUB_TOKEN=f157b23c5416f005713a6b28e5c46847f4ff838e
```

### MLflow Dockerfile
- Enhanced with `s3fs` support for S3 operations
- Uses custom entrypoint script for proper initialization
- Automatically waits for PostgreSQL before starting

### docker-compose.yml Updates
- MLflow service now uses entrypoint script
- Environment variables properly propagated
- Removed dependency on MinIO (uses DagsHub S3 directly)
- Extended healthcheck (20 retries, 30s start period)

## Workflow

### Training Pipeline
1. Airflow DAG triggers training
2. `train-api` receives request via `/train` endpoint
3. Service initializes MLflow experiment
4. Training occurs with live metric tracking
5. Model and artifacts uploaded to DagsHub S3
6. MLflow registers model version

### Prediction Pipeline
1. `predict-api` starts up
2. Attempts to load model from MLflow registry
3. Falls back to disk if MLflow unavailable
4. Predictions served via `/predict-*` endpoints

### Monitoring
- Prometheus scrapes metrics from `/metrics` endpoints
- Train API metrics include preprocessing and training stats
- Preprocess-API removed from all monitoring configs
- Alerts automatically updated

## Quick Start

### 1. Start Services
```bash
docker-compose up -d
```

### 2. Check MLflow UI
Visit: `http://localhost:5000`

### 3. Check Grafana Dashboards
Visit: `http://localhost:3000` (admin/admin)

### 4. Trigger Training DAG
```bash
# Via Airflow UI
# Or via CLI:
airflow dags trigger train_model_dag_batch_iterative_85000
```

### 5. Monitor in DagsHub
Visit: `https://dagshub.com/zz75da/rakuten_z/experiments`

## Troubleshooting

### MLflow Can't Connect to DagsHub
- Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are correct
- Check network connectivity to `dagshub.com`
- Review `MLFLOW_TRACKING_URI` format

### Model Not Loading from Registry
- Check MLflow UI to see if model is registered
- Verify model is in `Production` stage
- Check logs in `predict-api` container

### Preprocessing Failures
- All preprocessing is now in `train-api`
- Check `/train` endpoint logs for issues
- Verify training data CSV files exist

### DAG Waiting on Removed Service
- Verified: `preprocess-api` removed from DAG
- If using custom DAGs, update to remove `wait_for_preprocess_api` task

## File Changes Summary

### Modified Files
1. `mlflow_docker/requirements.txt` - Added s3fs, urllib3
2. `mlflow_docker/Dockerfile` - Added entrypoint script support
3. `mlflow_docker/entrypoint.sh` - New file for proper startup
4. `docker-compose.yml` - Updated MLflow service configuration
5. `train-api/app.py` - Added MLflow initialization
6. `predict-api/app.py` - Added MLflow model loading
7. `airflow/dags/train_dag.py` - Removed preprocess-api sensor
8. `monitoring/prometheus.yml` - Removed preprocess-api job
9. `monitoring/alertmanager.yml` - Removed preprocess-api routing
10. `monitoring/alert-rules.yml` - Removed preprocess-api alerts

## Next Steps

1. ✓ MLflow integrated with DagsHub S3
2. ✓ Train API logs to MLflow
3. ✓ Predict API loads from MLflow registry
4. ✓ Monitoring updated
5. Next: Configure DVC to also use DagsHub
6. Next: Set up automated model promotion to Production stage
7. Next: Create alerting for failed training runs
