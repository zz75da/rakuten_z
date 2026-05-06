#!/bin/bash
set -e

# MLflow Entrypoint Script for DagsHub Integration

echo "=== MLflow Startup with DagsHub S3 Integration ==="

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
until pg_isready -h postgres -p 5432 -U mlflow_user -d mlflow 2>/dev/null; do
    sleep 2
    echo "  PostgreSQL not ready yet, retrying..."
done
echo "✓ PostgreSQL is ready"

# Configure AWS/S3 credentials for DagsHub
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-https://dagshub.com/api/v1/repo-buckets/s3}"

# Set MLflow backend and artifact store URIs
export BACKEND_STORE_URI="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/mlflow"
export ARTIFACT_ROOT="${MLFLOW_ARTIFACT_URI:-s3://mlflow-artifacts}"

echo "MLflow Configuration:"
echo "  Backend Store: postgresql"
echo "  Artifact Root: $ARTIFACT_ROOT"
echo "  S3 Endpoint: $MLFLOW_S3_ENDPOINT_URL"
echo "  Tracking URI: ${MLFLOW_TRACKING_URI:-http://mlflow:5000}"

# Start MLflow server with proper S3/DagsHub configuration
echo "Starting MLflow server..."
exec mlflow server \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2
