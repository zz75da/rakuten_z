from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.python import PythonOperator
import json
import requests


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# ---------------- Python Callables ---------------- #

def get_auth_token(**kwargs):
    """Authenticate against gate-api and push JWT token to XCom."""
    response = requests.post(
        "http://gate-api:5000/login",
        json={"username": "admin", "password": "admin_pass"},
        timeout=10
    )
    response.raise_for_status()
    token = response.json()["token"]
    kwargs["ti"].xcom_push(key="auth_token", value=token)
    return token

def run_training(**kwargs):
    """
    Trigger train-api for dataset (sample/train).
    Preprocessing included. Dataset choice controlled by DAG conf.
    """
    dag_conf = kwargs["dag_run"].conf or {}
    use_dev_images = dag_conf.get("use_dev_images", True)  # Default True if not provided

    # Pick dataset name based on mode
    dataset_name = "sample" if use_dev_images else "full"

    url = "http://train-api:5002/train"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {kwargs['ti'].xcom_pull(task_ids='get_auth_token', key='auth_token')}"
    }
    payload = {
        "dataset": dataset_name,
        "epochs": 10,
        "batch_size": 64,
        "experiment_name": f"train_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
        "enable_mlflow_tracking": True,
        "mlflow_tracking_uri": "http://mlflow:5000"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10800)  # 3 hours max
    response.raise_for_status()
    return f"Training + preprocessing completed for dataset={dataset_name}"

def push_training_metrics(**kwargs):
    """Push training metrics placeholder to Prometheus Pushgateway."""
    push_url = "http://pushgateway:9091/metrics/job/rakuten_mlops"
    metrics = {
        "training_accuracy": 0.90,
        "validation_accuracy": 0.87,
        "training_loss": 0.2,
        "validation_loss": 0.25,
        "epochs_completed": 10,
        "training_time_seconds": 10800
    }
    data = "\n".join([f"{k} {v}" for k, v in metrics.items()])
    r = requests.post(push_url, data=data)
    r.raise_for_status()
    return "Metrics pushed to Prometheus successfully"

# ---------------- DAG Definition ---------------- #

with DAG(
    dag_id="rakuten_multimodal_pipeline",
    default_args=default_args,
    description="End-to-end preprocessing + training pipeline with MLflow & Prometheus",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training", "production", "monitoring"],
) as dag:

    # Verify input data & images exist (lightweight check)
    check_data = BashOperator(
        task_id="check_training_data",
        bash_command=(
            "echo '=== Checking data availability ===' && "
            "ls -lh /opt/airflow/data/ && "
            "[ -d /opt/airflow/data/images/image_train ] || (echo 'Missing image_train dir' && exit 1) && "
            "[ -d /opt/airflow/data/images/image_sample ] || (echo 'Missing image_sample dir' && exit 1) && "
            "echo '✓ All expected data present'"
        ),
    )

    # Wait for APIs to be ready
    wait_for_gate_api = HttpSensor(
        task_id="wait_for_gate_api",
        http_conn_id="gate_api",
        endpoint="/health",
        method="GET",
        response_check=lambda r: r.status_code == 200,
        timeout=300,
        poke_interval=15,
        mode="reschedule",
    )

    wait_for_train_api = HttpSensor(
        task_id="wait_for_train_api",
        http_conn_id="train_api",
        endpoint="/health",
        method="GET",
        response_check=lambda r: r.status_code == 200,
        timeout=300,
        poke_interval=15,
        mode="reschedule",
    )

    # API Calls
    get_token = PythonOperator(
        task_id="get_auth_token",
        python_callable=get_auth_token,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )

    push_metrics = PythonOperator(
        task_id="push_metrics",
        python_callable=push_training_metrics,
    )

    success_message = BashOperator(
        task_id="success_message",
        bash_command=(
            "echo '=== Pipeline Completed Successfully ===' && "
            "echo 'Check MLflow at http://localhost:5000' && "
            "echo 'Grafana at http://localhost:3000' && "
            "echo 'Streamlit UI at http://localhost:8501'"
        ),
    )

    # ---------------- DAG Flow ---------------- #
    check_data >> wait_for_gate_api >> wait_for_train_api >> get_token >> train_task >> push_metrics >> success_message
