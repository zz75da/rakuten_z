from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.http.sensors.http import HttpSensor
import requests
import json
import os
import subprocess
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dagshub  


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# ---------------- Helper Functions ---------------- #

def setup_logger(task_name: str):
    """Setup a per-task logger writing to /opt/airflow/data/artifacts/logs"""
    log_dir = "/opt/airflow/data/artifacts/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{task_name}.log")
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def safe_dvc_pull(logger):
    """Attempt DVC pull, fallback to local if fails"""
    try:
        logger.info(" Pulling artifacts via DVC...")
        subprocess.run(["dvc", "pull", "--remote", "origin"], check=True)
        logger.info(" DVC pull succeeded")
    except Exception as e:
        logger.warning(f" DVC pull failed: {e}. Using local cached artifacts")

def get_auth_token(**kwargs):
    task_name = "get_auth_token"
    logger = setup_logger(task_name)
    logger.info(" Requesting auth token from gate-api")
    try:
        response = requests.post(
            "http://gate-api:5000/login",
            json={"username": "admin", "password": "admin_pass"},
            timeout=10
        )
        response.raise_for_status()
        token = response.json()["token"]
        kwargs["ti"].xcom_push(key="auth_token", value=token)
        logger.info(" Auth token received")
        return token
    except Exception as e:
        logger.error(f" Failed to get auth token: {e}")
        raise

def ensure_dagshub_token(logger):
    """
    Ensure Dagshub token is cached for MLflow logging inside Airflow workers.
    Uses DAGSHUB_TOKEN from env if available, writes it with add_app_token().
    """
    user = os.getenv("DAGSHUB_USER")
    token = os.getenv("DAGSHUB_TOKEN")
    cache = os.getenv("DAGSHUB_CLIENT_TOKENS_CACHE")

    if token:
        try:
            if cache:
                dagshub.auth.add_app_token(token, cache_location=cache)
            else:
                dagshub.auth.add_app_token(token)
            logger.info(" DAGSHUB_TOKEN cached with add_app_token()")
        except Exception as e:
            logger.warning(f" Failed to cache DAGSHUB_TOKEN: {e}")
    else:
        logger.warning(" No DAGSHUB_TOKEN in env, MLflow may fallback to local tracking")

def run_training(**kwargs):
    task_name = "train_model"
    logger = setup_logger(task_name)
    logger.info("🔹 Starting training task")

    ensure_dagshub_token(logger)
    safe_dvc_pull(logger)

    dag_conf = kwargs.get("dag_run").conf or {}
    use_dev_images = dag_conf.get("use_dev_images", True)
    dataset_name = "sample" if use_dev_images else "full"

    url = "http://train-api:5002/train"
    token = kwargs['ti'].xcom_pull(task_ids='get_auth_token', key='auth_token')
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {
        "use_dev_images": use_dev_images,
        "epochs": dag_conf.get("epochs", 10),
        "batch_size": dag_conf.get("batch_size", 64)
    }

    logger.info(f"📡 Sending training request to {url} with payload: {payload}")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10800)
        response.raise_for_status()
        result = response.json()
        logger.info(f" Training completed, MLflow run_id: {result.get('mlflow_run_id')}")
    except Exception as e:
        logger.error(f" Training failed: {e}")
        raise

    artifacts_dir = "/opt/airflow/data/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    history_path = os.path.join(artifacts_dir, f"train_history_{dataset_name}.json")
    with open(history_path, "w") as f:
        json.dump(result.get("history", {}), f)
    kwargs['ti'].xcom_push(key="train_history_path", value=history_path)

    for key in ["final_metrics", "train_params", "mlflow_run_id", "mlflow_model_version", "model_path"]:
        kwargs['ti'].xcom_push(key=key, value=result.get(key))

    return result

def run_prediction(**kwargs):
    task_name = "predict_model"
    logger = setup_logger(task_name)
    logger.info("🔹 Running prediction task")

    safe_dvc_pull(logger)
    token = kwargs['ti'].xcom_pull(task_ids='get_auth_token', key='auth_token')
    headers = {"Authorization": f"Bearer {token}"}
    test_images_dir = "/opt/airflow/data/images/image_test"
    payload = {"image_dir": test_images_dir}
    url = "http://predict-api:5003/predict"

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        results = response.json()
        logger.info(f" Prediction completed, {len(results)} outputs")
    except Exception as e:
        logger.error(f" Prediction failed: {e}")
        results = {"error": str(e)}

    artifacts_dir = "/opt/airflow/data/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    pred_path = os.path.join(artifacts_dir, "prediction_results.json")
    with open(pred_path, "w") as f:
        json.dump(results, f)
    kwargs['ti'].xcom_push(key="prediction_path", value=pred_path)
    return results

def plot_training_curve(**kwargs):
    task_name = "plot_training_curve"
    logger = setup_logger(task_name)
    logger.info("📊 Generating training curve plot")

    ti = kwargs['ti']
    history_path = ti.xcom_pull(task_ids="train_model", key="train_history_path")
    if not history_path or not os.path.exists(history_path):
        logger.warning("⚠️ Training history file not found, skipping plot")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    artifacts_dir = "/opt/airflow/data/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, "training_curve.png")

    plt.figure(figsize=(8,5))
    if "loss" in history: plt.plot(history["loss"], label="Train Loss")
    if "val_loss" in history: plt.plot(history.get("val_loss", []), label="Val Loss")
    if "accuracy" in history: plt.plot(history.get("accuracy", []), label="Train Acc")
    if "val_accuracy" in history: plt.plot(history.get("val_accuracy", []), label="Val Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    logger.info(f" Training curve saved at {out_path}")
    ti.xcom_push(key="training_curve_path", value=out_path)

def push_training_summary(**kwargs):
    task_name = "push_summary"
    logger = setup_logger(task_name)
    logger.info(" Pushing training summary")

    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids="train_model", key="final_metrics") or {}
    params = ti.xcom_pull(task_ids="train_model", key="train_params") or {}
    run_id = ti.xcom_pull(task_ids="train_model", key="mlflow_run_id")
    model_version = ti.xcom_pull(task_ids="train_model", key="mlflow_model_version")

    logger.info(f"MLflow run_id: {run_id}, model_version: {model_version}")
    logger.info(f"Params: {params}")
    logger.info(f"Metrics: {metrics}")

    push_url = "http://pushgateway:9091/metrics/job/rakuten_mlops"
    if metrics:
        # flatten metrics for Prometheus (final_accuracy + f1 scores)
        lines = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                lines.append(f"{k} {v}")
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if isinstance(subv, (int, float)):
                        lines.append(f"{k}_{subk} {subv}")
        data = "\n".join(lines)
        try:
            requests.post(push_url, data=data).raise_for_status()
            logger.info(" Metrics pushed to Prometheus Pushgateway")
        except Exception as e:
            logger.warning(f" Failed to push metrics to Prometheus: {e}")

    return {"mlflow_run_id": run_id, "mlflow_model_version": model_version, "metrics": metrics}

# ---------------- DAG Definition ---------------- #

with DAG(
    dag_id="rakuten_multimodal_pipeline_v1_7",
    default_args=default_args,
    description="Full training + prediction pipeline with DVC, MLflow, Prometheus, fail-safe, verbose logging",
    schedule_interval=None,
    start_date=datetime(2024,1,1),
    catchup=False,
    tags=["mlops","training","prediction","tracking","fail_safe","logging"]
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

    wait_gate_api = HttpSensor(
        task_id="wait_for_gate_api",
        http_conn_id="gate_api",
        endpoint="/health",
        method="GET",
        response_check=lambda r: r.status_code == 200,
        timeout=300,
        poke_interval=15,
        mode="reschedule",
    )

    wait_train_api = HttpSensor(
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

    predict_task = PythonOperator(
        task_id="predict_model",
        python_callable=run_prediction,
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
    (
        check_data
        >> [wait_gate_api, wait_train_api]
        >> get_token
        >> train_task
        >> predict_task
        >> plot_task
        >> push_summary
        >> success_message
    )
