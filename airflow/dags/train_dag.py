from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.python import PythonOperator
import json
import time
import requests
import pandas as pd
import os

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

TRAIN_API = "http://train-api:5002"
GATE_API  = "http://gate-api:5000"

# Maximum time (seconds) to wait for training to finish: 4 hours
TRAINING_MAX_WAIT = int(os.environ.get("TRAINING_MAX_WAIT_SECONDS", 4 * 3600))
TRAINING_POLL_INTERVAL = int(os.environ.get("TRAINING_POLL_INTERVAL_SECONDS", 60))


# --- Python Callables ---

def get_auth_token(**context):
    response = requests.post(
        f"{GATE_API}/login",
        json={"username": "admin", "password": "admin_pass"},
        timeout=10,
    )
    if response.status_code != 200:
        raise Exception(f"Login failed: {response.text}")
    token = response.json()["token"]
    context["ti"].xcom_push(key="auth_token", value=token)
    return token


def check_dataset_stats(**context):
    """Read CSV metadata and push shape info to XCom (no training logic here)."""
    csv_path = "/opt/airflow/data/X_train_update.csv"
    df = pd.read_csv(csv_path, nrows=0)   # headers only — fast
    full = pd.read_csv(csv_path)
    shape = f"{len(full)}x{len(full.columns)}"
    context["ti"].xcom_push(key="data_shape", value=shape)
    context["ti"].xcom_push(key="data_columns", value=list(full.columns))
    print(f"Dataset: {shape}, columns: {list(full.columns)}")
    return shape


def trigger_and_poll_training(**context):
    """
    1. POST /train  → get job_id (returns 202 immediately).
    2. Poll GET /train/status/{job_id} every TRAINING_POLL_INTERVAL seconds.
    3. Return result dict when status == 'success', raise on 'failed'.
    """
    token = context["ti"].xcom_pull(task_ids="get_auth_token", key="auth_token")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # 1. Trigger
    payload = {
        "use_dev_images": True,
        "epochs": int(os.environ.get("TRAIN_EPOCHS", 10)),
        "batch_size": int(os.environ.get("TRAIN_BATCH_SIZE", 32)),
    }
    resp = requests.post(f"{TRAIN_API}/train", json=payload, headers=headers, timeout=900)
    if not resp.ok:
        raise Exception(
            f"POST /train returned HTTP {resp.status_code}. Body: {resp.text[:2000]}"
        )
    job_id = resp.json()["job_id"]
    print(f"Training job started: {job_id}")
    context["ti"].xcom_push(key="training_job_id", value=job_id)

    # 2. Poll — resilient to transient timeouts under high CPU load
    elapsed = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10  # fail only after 10 consecutive unreachable polls

    while elapsed < TRAINING_MAX_WAIT:
        time.sleep(TRAINING_POLL_INTERVAL)
        elapsed += TRAINING_POLL_INTERVAL

        try:
            status_resp = requests.get(
                f"{TRAIN_API}/train/status/{job_id}",
                headers=headers,
                timeout=60,  # generous: server may be CPU-bound during training
            )
            status_resp.raise_for_status()
            consecutive_failures = 0
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            consecutive_failures += 1
            print(
                f"[{elapsed}s] Status poll failed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {exc}"
            )
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise Exception(
                    f"Status endpoint unreachable {MAX_CONSECUTIVE_FAILURES} times in a row — aborting"
                )
            continue  # skip this tick, retry next interval

        job = status_resp.json()
        status = job.get("status")
        print(f"[{elapsed}s] Job {job_id} status: {status}")

        if status == "success":
            context["ti"].xcom_push(key="training_result", value=job)
            return job
        if status == "failed":
            raise Exception(f"Training job {job_id} failed: {job.get('error')}")
        # status == "running" → keep polling

    raise Exception(
        f"Training job {job_id} did not finish within {TRAINING_MAX_WAIT // 3600}h"
    )


def get_model_version(**context):
    """Retrieve the latest model version from DagsHub MLflow registry."""
    try:
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        dagshub_user = os.environ.get("DAGSHUB_USER", "")
        dagshub_token = os.environ.get("DAGSHUB_TOKEN", "")
        model_name = os.environ.get("MLFLOW_MODEL_NAME", "rakuten_multimodal")

        if not mlflow_uri or not dagshub_user:
            print("MLFLOW_TRACKING_URI / DAGSHUB_USER not set — skipping version check")
            return "unknown"

        # DagsHub MLflow REST API
        api_url = (
            f"https://dagshub.com/{dagshub_user}/rakuten_z.mlflow"
            f"/api/2.0/mlflow/registered-models/get-latest-versions"
        )
        resp = requests.get(
            api_url,
            params={"name": model_name, "stages": ["Staging", "Production"]},
            auth=(dagshub_user, dagshub_token),
            timeout=20,
        )
        if resp.status_code == 200:
            versions = resp.json().get("model_versions", [])
            if versions:
                version = versions[0]["version"]
                context["ti"].xcom_push(key="model_version", value=version)
                print(f"Latest model version: {version}")
                return version
        print(f"Version check returned {resp.status_code}: {resp.text[:200]}")
        return "unknown"
    except Exception as exc:
        print(f"Version check failed (non-blocking): {exc}")
        return "unknown"


def push_training_metrics(**context):
    """Push real training metrics from the completed job to Prometheus Pushgateway."""
    result = context["ti"].xcom_pull(task_ids="train_model", key="training_result") or {}
    final = result.get("final_metrics", {})
    history = result.get("history", {})

    accuracy     = final.get("accuracy", 0)
    val_accuracy = history.get("val_accuracy", [0])[-1]
    loss         = history.get("loss", [0])[-1]
    val_loss     = history.get("val_loss", [0])[-1]
    epochs_done  = len(history.get("loss", []))

    started_at = result.get("started_at", "")
    completed_at = result.get("completed_at", "")
    if started_at and completed_at:
        from dateutil import parser as dtparser
        elapsed = (dtparser.parse(completed_at) - dtparser.parse(started_at)).total_seconds()
    else:
        elapsed = 0

    metrics = {
        "training_accuracy": accuracy,
        "validation_accuracy": val_accuracy,
        "training_loss": loss,
        "validation_loss": val_loss,
        "epochs_completed": epochs_done,
        "training_time_seconds": elapsed,
    }
    metrics_text = "\n".join(f"{k} {v}" for k, v in metrics.items())

    try:
        resp = requests.post(
            "http://pushgateway:9091/metrics/job/rakuten_mlops",
            data=metrics_text,
            timeout=10,
        )
        print(f"Metrics pushed (HTTP {resp.status_code}): {metrics}")
    except Exception as exc:
        print(f"Prometheus push failed (non-blocking): {exc}")


def evaluate_from_result(**context):
    """Log final evaluation metrics from training result stored in XCom."""
    result = context["ti"].xcom_pull(task_ids="train_model", key="training_result") or {}
    final = result.get("final_metrics", {})
    history = result.get("history", {})

    print("=== Model Evaluation Results ===")
    print(f"Final accuracy : {final.get('accuracy', 'N/A'):.4f}" if final.get("accuracy") else "accuracy: N/A")
    val_accs = history.get("val_accuracy", [])
    if val_accs:
        print(f"Best val_accuracy: {max(val_accs):.4f}  (last: {val_accs[-1]:.4f})")
    mlflow_run = result.get("mlflow_run_id", "N/A")
    print(f"MLflow run id  : {mlflow_run}")
    return final


# --- DAG Definition ---
with DAG(
    dag_id="rakuten_multimodal_pipeline_v5_1",
    default_args=default_args,
    description="Train multimodal model (async) with MLflow & Prometheus",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training", "production", "monitoring"],
) as dag:

    check_data = BashOperator(
        task_id="check_training_data",
        bash_command=(
            "echo '=== Checking training data ===' && "
            "[ -f /opt/airflow/data/X_train_update.csv ] || (echo 'Missing X_train_update.csv' && exit 1) && "
            "[ -f /opt/airflow/data/Y_train_CVw08PX.csv ] || (echo 'Missing Y_train_CVw08PX.csv' && exit 1) && "
            "[ -d /opt/airflow/data/images/image_train/ ] || (echo 'Missing training images' && exit 1) && "
            "wc -l /opt/airflow/data/X_train_update.csv && "
            "echo '✓ All training files present'"
        ),
    )

    wait_for_gate_api = HttpSensor(
        task_id="wait_for_gate_api",
        http_conn_id="gate_api",
        endpoint="/health",
        method="GET",
        response_check=lambda response: response.status_code == 200,
        timeout=300,
        poke_interval=15,
        mode="reschedule",
    )

    wait_for_train_api = HttpSensor(
        task_id="wait_for_train_api",
        http_conn_id="train_api",
        endpoint="/health",
        method="GET",
        response_check=lambda response: response.status_code == 200,
        timeout=300,
        poke_interval=15,
        mode="reschedule",
    )

    wait_for_predict_api = HttpSensor(
        task_id="wait_for_predict_api",
        http_conn_id="predict_api",
        endpoint="/health",
        method="GET",
        response_check=lambda response: response.status_code == 200,
        timeout=300,
        poke_interval=15,
        mode="reschedule",
    )

    dataset_stats = PythonOperator(
        task_id="dataset_stats",
        python_callable=check_dataset_stats,
        provide_context=True,
    )

    get_token = PythonOperator(
        task_id="get_auth_token",
        python_callable=get_auth_token,
        provide_context=True,
    )

    # execution_timeout must be > TRAINING_MAX_WAIT to avoid Airflow killing the task
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=trigger_and_poll_training,
        provide_context=True,
        execution_timeout=timedelta(seconds=TRAINING_MAX_WAIT + 600),
    )

    get_version = PythonOperator(
        task_id="get_model_version",
        python_callable=get_model_version,
        provide_context=True,
    )

    push_metrics = PythonOperator(
        task_id="push_metrics",
        python_callable=push_training_metrics,
        provide_context=True,
    )

    eval_results = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_from_result,
        provide_context=True,
    )

    verify_mlflow = BashOperator(
        task_id="verify_mlflow_registration",
        bash_command=(
            "echo '=== Verifying MLflow registration on DagsHub ===' && "
            "DAGSHUB_USER=${DAGSHUB_USER:-} && "
            "DAGSHUB_TOKEN=${DAGSHUB_TOKEN:-} && "
            "MODEL=rakuten_multimodal && "
            "if [ -z \"$DAGSHUB_USER\" ]; then echo 'DAGSHUB_USER not set, skipping'; exit 0; fi && "
            "URL=\"https://dagshub.com/${DAGSHUB_USER}/rakuten_z.mlflow/api/2.0/mlflow/registered-models/get?name=${MODEL}\" && "
            "response=$(curl -s -u \"${DAGSHUB_USER}:${DAGSHUB_TOKEN}\" \"$URL\") && "
            "if echo \"$response\" | grep -q '\"name\"'; then "
            "    echo \"✓ Model '${MODEL}' registered in DagsHub MLflow\"; "
            "else "
            "    echo \"Model not found: $response\"; exit 1; "
            "fi"
        ),
    )

    success_message = BashOperator(
        task_id="success_message",
        bash_command=(
            "echo '=== MLOps Pipeline Completed ===' && "
            "echo 'Grafana : http://localhost:3000' && "
            "echo 'Streamlit: http://localhost:8501' && "
            "echo 'Pipeline executed: {{ ds }}'"
        ),
    )

    # --- Task Dependencies ---
    (
        check_data
        >> [wait_for_gate_api, wait_for_train_api, wait_for_predict_api]
        >> dataset_stats
        >> get_token
        >> train_model_task
        >> [get_version, push_metrics]
        >> eval_results
        >> verify_mlflow
        >> success_message
    )
