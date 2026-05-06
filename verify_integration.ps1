#!/usr/bin/env pwsh
# Verification script for MLflow + DagsHub integration

Write-Host "=== Rakuten MLOps - MLflow + DagsHub Integration Verification ===" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "1. Checking Docker daemon..." -ForegroundColor Yellow
try {
    $docker = docker ps --format "table {{.Names}}" 2>$null | Select-Object -First 1
    if ($docker) {
        Write-Host "   ✓ Docker is running" -ForegroundColor Green
    } else {
        Write-Host "   ✗ Docker is not running" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "   ✗ Docker not found" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "2. Configuration Files Status:" -ForegroundColor Yellow

$files = @(
    "docker-compose.yml",
    "mlflow_docker/Dockerfile",
    "mlflow_docker/requirements.txt",
    "mlflow_docker/entrypoint.sh",
    ".env",
    "train-api/app.py",
    "predict-api/app.py",
    "airflow/dags/train_dag.py",
    "monitoring/prometheus.yml",
    "monitoring/alertmanager.yml",
    "monitoring/alert-rules.yml"
)

foreach ($file in $files) {
    $path = Join-Path "c:\Users\zobir\DScientest\rakuten_mlops_services" $file
    if (Test-Path $path) {
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $file (missing)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "3. Environment Configuration:" -ForegroundColor Yellow

$envPath = "c:\Users\zobir\DScientest\rakuten_mlops_services\.env"
if (Test-Path $envPath) {
    $env_content = Get-Content $envPath | Select-String "MLFLOW|DAGSHUB|AWS"
    if ($env_content) {
        Write-Host "   ✓ DagsHub credentials found in .env" -ForegroundColor Green
        Write-Host "     - MLflow Configuration: $($env_content | Where-Object {$_ -match 'MLFLOW'} | Measure-Object).Count entries"
        Write-Host "     - DagsHub Configuration: $($env_content | Where-Object {$_ -match 'DAGSHUB'} | Measure-Object).Count entries"
    } else {
        Write-Host "   ✗ DagsHub credentials not found" -ForegroundColor Red
    }
} else {
    Write-Host "   ✗ .env file not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "4. Verification of Key Changes:" -ForegroundColor Yellow

# Check if preprocess-api is removed from DAG
$dagPath = "c:\Users\zobir\DScientest\rakuten_mlops_services\airflow\dags\train_dag.py"
$dagContent = Get-Content $dagPath -Raw
if ($dagContent -notmatch 'wait_for_preprocess_api') {
    Write-Host "   ✓ preprocess-api removed from DAG" -ForegroundColor Green
} else {
    Write-Host "   ✗ preprocess-api still referenced in DAG" -ForegroundColor Red
}

# Check if MLflow is configured in train-api
$trainAppPath = "c:\Users\zobir\DScientest\rakuten_mlops_services\train-api\app.py"
$trainAppContent = Get-Content $trainAppPath -Raw
if ($trainAppContent -match 'mlflow.set_tracking_uri|MLFLOW_TRACKING_URI') {
    Write-Host "   ✓ MLflow tracking initialized in train-api" -ForegroundColor Green
} else {
    Write-Host "   ✗ MLflow not configured in train-api" -ForegroundColor Red
}

# Check if predict-api loads from MLflow
$predictAppPath = "c:\Users\zobir\DScientest\rakuten_mlops_services\predict-api\app.py"
$predictAppContent = Get-Content $predictAppPath -Raw
if ($predictAppContent -match 'mlflow.keras.load_model|MlflowClient') {
    Write-Host "   ✓ MLflow model loading in predict-api" -ForegroundColor Green
} else {
    Write-Host "   ✗ MLflow model loading not configured in predict-api" -ForegroundColor Red
}

# Check if prometheus config is clean
$promPath = "c:\Users\zobir\DScientest\rakuten_mlops_services\monitoring\prometheus.yml"
$promContent = Get-Content $promPath -Raw
if ($promContent -notmatch 'preprocess-api') {
    Write-Host "   ✓ preprocess-api removed from Prometheus config" -ForegroundColor Green
} else {
    Write-Host "   ✗ preprocess-api still in Prometheus config" -ForegroundColor Red
}

Write-Host ""
Write-Host "5. Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Start services: docker-compose up -d" -ForegroundColor Cyan
Write-Host "   2. Wait for services to be healthy (~60 seconds)" -ForegroundColor Cyan
Write-Host "   3. Access MLflow UI: http://localhost:5000" -ForegroundColor Cyan
Write-Host "   4. Trigger training DAG from Airflow: http://localhost:8080" -ForegroundColor Cyan
Write-Host "   5. Monitor experiments on DagsHub: https://dagshub.com/zz75da/rakuten_z" -ForegroundColor Cyan
Write-Host "   6. View Grafana dashboards: http://localhost:3000" -ForegroundColor Cyan

Write-Host ""
Write-Host "✓ Integration verification complete!" -ForegroundColor Green
