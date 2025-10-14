from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.python import PythonOperator
import requests
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    """Authenticate with gate-api and push JWT token to XCom"""
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
    """Trigger training via train-api, save artifacts, and push outputs to XCom"""
    dag_conf = kwargs.get("dag_run").conf or {}
    use_dev_images = dag_conf.get("use_dev_images", True)
    dataset_name = "sample" if use_dev_images else "full"

    print(f"🔹 Running training in mode: {'DEV' if use_dev_images else 'FULL TRAIN'}")
    
    url = "http://train-api:5002/train"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {kwargs['ti'].xcom_pull(task_ids='get_auth_token', key='auth_token')}"
    }
    payload = {
        "use_dev_images": use_dev_images,
        "epochs": dag_conf.get("epochs", 10),
        "batch_size": dag_conf.get("batch_size", 64)
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10800)
    response.raise_for_status()
    result = response.json()

    artifacts_dir = "/opt/airflow/data/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save training history
    history_path = os.path.join(artifacts_dir, f"train_history_{dataset_name}.json")
    with open(history_path, "w") as f:
        json.dump(result.get("history", {}), f)
    kwargs['ti'].xcom_push(key="train_history_path", value=history_path)

    # Push key outputs to XCom
    kwargs['ti'].xcom_push(key="final_metrics", value=result.get("final_metrics", {}))
    kwargs['ti'].xcom_push(key="train_params", value=payload)
    kwargs['ti'].xcom_push(key="mlflow_run_id", value=result.get("mlflow_run_id"))
    kwargs['ti'].xcom_push(key="mlflow_model_version", value=result.get("mlflow_model_version"))
    kwargs['ti'].xcom_push(key="model_path", value=result.get("model_path"))

    return result


def plot_training_curve(**kwargs):
    """Plot training curves (loss/accuracy) from saved history"""
    ti = kwargs['ti']
    history_path = ti.xcom_pull(task_ids="train_model", key="train_history_path")
    if not history_path or not os.path.exists(history_path):
        raise ValueError("Training history file not found")

    with open(history_path, "r") as f:
        history = json.load(f)

    artifacts_dir = "/opt/airflow/data/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, "training_curve.png")

    plt.figure(figsize=(8, 5))
    if "loss" in history: plt.plot(history["loss"], label="Train Loss")
    if "val_loss" in history: plt.plot(history["val_loss"], label="Val Loss")
    if "accuracy" in history: plt.plot(history["accuracy"], label="Train Acc")
    if "val_accuracy" in history: plt.plot(history["val_accuracy"], label="Val Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Progress ({'DEV' if 'sample' in history_path else 'FULL TRAIN'})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)

    ti.xcom_push(key="training_curve_path", value=out_path)


def push_training_summary(**kwargs):
    """Log summary of training, push metrics to Prometheus Pushgateway"""
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids="train_model", key="final_metrics") or {}
    params = ti.xcom_pull(task_ids="train_model", key="train_params") or {}
    run_id = ti.xcom_pull(task_ids="train_model", key="mlflow_run_id")
    model_version = ti.xcom_pull(task_ids="train_model", key="mlflow_model_version")

    print("=== Training Summary ===")
    print(f"MLflow run_id: {run_id}")
    print(f"MLflow model_version: {model_version}")
    print(f"Params: {params}")
    print(f"Metrics: {metrics}")

    # Push key metrics to Prometheus Pushgateway (optional)
    push_url = "http://pushgateway:9091/metrics/job/rakuten_mlops"
    if metrics:
        data = "\n".join([f"{k} {v}" for k, v in metrics.items()])
        try:
            requests.post(push_url, data=data).raise_for_status()
            print(" Metrics pushed to Prometheus Pushgateway")
        except Exception as e:
            print(f" Failed to push metrics to Prometheus: {e}")

    return {
        "mlflow_run_id": run_id,
        "mlflow_model_version": model_version,
        "metrics": metrics
    }


# ---------------- DAG Definition ---------------- #

with DAG(
    dag_id="rakuten_multimodal_pipeline_full_tracking_02",
    default_args=default_args,
    description="Full training pipeline consuming train-api MLflow logging + registry",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training", "production", "tracking"],
) as dag:

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

    get_token = PythonOperator(
        task_id="get_auth_token",
        python_callable=get_auth_token,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )

    plot_task = PythonOperator(
        task_id="plot_training_curve",
        python_callable=plot_training_curve,
    )

    push_summary = PythonOperator(
        task_id="push_summary",
        python_callable=push_training_summary,
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
    check_data >> wait_for_gate_api >> wait_for_train_api >> get_token
    get_token >> train_task >> plot_task >> push_summary >> success_message
