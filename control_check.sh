#!/bin/bash
# control_check.sh — run from project root

echo "=== 1️⃣ Checking Airflow -> Postgres connectivity ==="
docker-compose exec airflow-webserver bash -c "
echo \$AIRFLOW__CORE__SQL_ALCHEMY_CONN;
ping -c 2 postgres;
psql -h postgres -U airflow -d airflow -c '\l'
"

echo -e "\n=== 2️⃣ Checking Train API health ==="
docker-compose exec airflow-webserver bash -c "
curl -s -o /dev/null -w '%{http_code} %{url_effective}\n' http://train-api:5002/health;
curl -s -o /dev/null -w '%{http_code} %{url_effective}\n' http://train-api:5002/metrics;
"

echo -e "\n=== 3️⃣ Checking MLflow artifact upload ==="
docker-compose exec airflow-webserver bash -c "
echo \$MLFLOW_TRACKING_URI;
echo \$MLFLOW_ARTIFACT_URI;
mlflow artifacts list --run-id <YOUR_RUN_ID> --artifact-path <ARTIFACT_PATH>