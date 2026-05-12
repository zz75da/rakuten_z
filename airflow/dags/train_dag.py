from datetime import datetime, timedelta
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.exceptions import AirflowException
import requests
import pandas as pd
import os

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

TRAIN_API        = "http://train-api:5002"
GATE_API         = "http://gate-api:5000"
MINILM_ENCODER   = "http://minilm-encoder:5004"

TRAINING_MAX_WAIT = int(os.environ.get("TRAINING_MAX_WAIT_SECONDS", 30 * 3600))

# Airflow Variable keys — one per training job so they don't collide
_CV_JOB_ID_VAR     = "rakuten_cv_training_job_id"
_MINILM_JOB_ID_VAR = "rakuten_minilm_training_job_id"


# --- Helpers ---

def _fresh_token():
    resp = requests.post(
        f"{GATE_API}/login",
        json={"username": "admin", "password": "admin_pass"},
        timeout=10,
    )
    if resp.status_code != 200:
        raise Exception(f"Login failed: {resp.text}")
    return resp.json()["token"]


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# --- Python Callables ---

def get_auth_token(**context):
    token = _fresh_token()
    context["ti"].xcom_push(key="auth_token", value=token)
    return token


def check_dataset_stats(**context):
    csv_path = "/opt/airflow/data/X_train_update.csv"
    full = pd.read_csv(csv_path)
    shape = f"{len(full)}x{len(full.columns)}"
    context["ti"].xcom_push(key="data_shape", value=shape)
    context["ti"].xcom_push(key="data_columns", value=list(full.columns))
    print(f"Dataset: {shape}, columns: {list(full.columns)}")
    return shape


def _trigger_training_job(job_var_key, text_encoder, context):
    """
    Shared logic for triggering a training job (CV or MiniLM).
    Resumes an existing job if one is stored in the Variable.
    """
    token = _fresh_token()

    existing_job_id = Variable.get(job_var_key, default_var=None)
    if existing_job_id:
        try:
            probe = requests.get(
                f"{TRAIN_API}/train/status/{existing_job_id}",
                headers=_auth_headers(token), timeout=120,
            )
            if probe.status_code == 404:
                print(f"Job {existing_job_id} gone (train-api restarted). Starting fresh.")
                Variable.delete(job_var_key)
                existing_job_id = None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            print(f"Probe timed out for {existing_job_id} — assuming still running.")
        if existing_job_id:
            print(f"Resuming existing {text_encoder} job: {existing_job_id}")
            context["ti"].xcom_push(key=f"{text_encoder}_job_id", value=existing_job_id)
            return existing_job_id

    payload = {
        "use_dev_images": False,
        "epochs": int(os.environ.get("TRAIN_EPOCHS", 30)),
        "batch_size": int(os.environ.get("TRAIN_BATCH_SIZE", 128)),
        "text_encoder": text_encoder,
    }
    resp = requests.post(
        f"{TRAIN_API}/train", json=payload,
        headers=_auth_headers(token), timeout=30,
    )
    if resp.status_code == 401:
        token = _fresh_token()
        resp = requests.post(
            f"{TRAIN_API}/train", json=payload,
            headers=_auth_headers(token), timeout=30,
        )
    if not resp.ok:
        raise Exception(f"POST /train returned HTTP {resp.status_code}. Body: {resp.text[:2000]}")
    job_id = resp.json()["job_id"]
    print(f"{text_encoder} training job started: {job_id}")
    Variable.set(job_var_key, job_id)
    context["ti"].xcom_push(key=f"{text_encoder}_job_id", value=job_id)
    return job_id


def trigger_cv_training(**context):
    return _trigger_training_job(_CV_JOB_ID_VAR, "countvectorizer", context)


def trigger_minilm_encoding(**context):
    """Call POST /encode on the minilm-encoder service."""
    resp = requests.post(f"{MINILM_ENCODER}/encode", timeout=30)
    data = resp.json()
    print(f"MiniLM encoder: {data}")
    if data.get("status") == "error":
        raise AirflowException(f"MiniLM encoding error: {data.get('message')}")


def trigger_minilm_training(**context):
    return _trigger_training_job(_MINILM_JOB_ID_VAR, "minilm", context)


class TrainingCompleteSensor(BaseSensorOperator):
    """
    Polls /train/status/{job_id} using mode='reschedule'.
    job_var_key selects which Airflow Variable holds the active job ID.
    """

    def __init__(self, job_var_key, **kwargs):
        super().__init__(**kwargs)
        self.job_var_key = job_var_key

    def poke(self, context):
        job_id = Variable.get(self.job_var_key, default_var=None)
        if not job_id:
            raise AirflowException(
                f"No training job_id in Variable '{self.job_var_key}' — re-trigger the DAG"
            )

        token = _fresh_token()
        try:
            resp = requests.get(
                f"{TRAIN_API}/train/status/{job_id}",
                headers=_auth_headers(token), timeout=120,
            )
            if resp.status_code == 401:
                token = _fresh_token()
                resp = requests.get(
                    f"{TRAIN_API}/train/status/{job_id}",
                    headers=_auth_headers(token), timeout=120,
                )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            print(f"Status check timed out/unreachable (will retry in 5 min): {exc}")
            return False

        if resp.status_code == 404:
            Variable.delete(self.job_var_key)
            raise AirflowException(
                f"Job {job_id} not found (train-api restarted). "
                "Clear the DAG run and re-trigger to start a new job."
            )
        resp.raise_for_status()

        job   = resp.json()
        status = job.get("status")
        print(f"Job {job_id} [{self.job_var_key}]: status={status}")

        if status == "success":
            Variable.delete(self.job_var_key)
            context["ti"].xcom_push(key="training_result", value=job)
            return True
        if status == "failed":
            Variable.delete(self.job_var_key)
            raise AirflowException(f"Training job {job_id} failed: {job.get('error')}")
        return False


class MiniLMEncodingSensor(BaseSensorOperator):
    """
    Polls GET /status on the minilm-encoder service until encoding is done.
    """

    def poke(self, context):
        try:
            resp = requests.get(f"{MINILM_ENCODER}/status", timeout=30)
            data   = resp.json()
            status = data.get("status")
            print(f"MiniLM encoder status={status}: {data.get('message', '')}")
            if status == "done":
                return True
            if status == "error":
                raise AirflowException(f"MiniLM encoding failed: {data.get('message')}")
            return False
        except AirflowException:
            raise
        except Exception as exc:
            print(f"MiniLM status check failed (retrying): {exc}")
            return False


def get_model_version(**context):
    try:
        mlflow_uri   = os.environ.get("MLFLOW_TRACKING_URI", "")
        dagshub_user = os.environ.get("DAGSHUB_USER", "")
        dagshub_token = os.environ.get("DAGSHUB_TOKEN", "")
        model_name   = os.environ.get("MLFLOW_MODEL_NAME", "rakuten_multimodal")

        if not mlflow_uri or not dagshub_user:
            print("MLFLOW_TRACKING_URI / DAGSHUB_USER not set — skipping version check")
            return "unknown"

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
    """Push CV training metrics to Prometheus Pushgateway."""
    result = context["ti"].xcom_pull(task_ids="wait_for_cv_training", key="training_result") or {}
    final  = result.get("final_metrics", {})
    history = result.get("history", {})

    accuracy     = final.get("accuracy", 0)
    val_accuracy = history.get("val_accuracy", [0])[-1]
    loss         = history.get("loss", [0])[-1]
    val_loss     = history.get("val_loss", [0])[-1]
    epochs_done  = len(history.get("loss", []))

    started_at   = result.get("started_at", "")
    completed_at = result.get("completed_at", "")
    if started_at and completed_at:
        from dateutil import parser as dtparser
        elapsed = (dtparser.parse(completed_at) - dtparser.parse(started_at)).total_seconds()
    else:
        elapsed = 0

    metrics = {
        "training_accuracy":       accuracy,
        "validation_accuracy":     val_accuracy,
        "training_loss":           loss,
        "validation_loss":         val_loss,
        "epochs_completed":        epochs_done,
        "training_time_seconds":   elapsed,
    }
    metrics_text = "\n".join(f"{k} {v}" for k, v in metrics.items())
    try:
        resp = requests.post(
            "http://pushgateway:9091/metrics/job/rakuten_mlops",
            data=metrics_text, timeout=10,
        )
        print(f"CV metrics pushed (HTTP {resp.status_code}): {metrics}")
    except Exception as exc:
        print(f"Prometheus push failed (non-blocking): {exc}")


def evaluate_from_result(**context):
    """Log evaluation results for both CV and MiniLM training runs."""
    for task_id, label in [
        ("wait_for_cv_training",    "CountVectorizer"),
        ("wait_for_minilm_training", "MiniLM"),
    ]:
        result = context["ti"].xcom_pull(task_ids=task_id, key="training_result") or {}
        final   = result.get("final_metrics", {})
        history = result.get("history", {})
        print(f"\n=== {label} Model Results ===")
        if final.get("accuracy"):
            print(f"  Final accuracy   : {final['accuracy']:.4f}")
        val_accs = history.get("val_accuracy", [])
        if val_accs:
            print(f"  Best val_accuracy: {max(val_accs):.4f}  (last: {val_accs[-1]:.4f})")
        print(f"  MLflow run id    : {result.get('mlflow_run_id', 'N/A')}")
    return {}


# --- DAG Definition ---
with DAG(
    dag_id="rakuten_multimodal_pipeline_v5_1",
    default_args=default_args,
    description="Sequential CV + MiniLM dual-model training with MLflow & Prometheus",
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
        timeout=300, poke_interval=15, mode="reschedule",
    )

    wait_for_train_api = HttpSensor(
        task_id="wait_for_train_api",
        http_conn_id="train_api",
        endpoint="/health",
        method="GET",
        response_check=lambda response: response.status_code == 200,
        timeout=300, poke_interval=15, mode="reschedule",
    )

    wait_for_predict_api = HttpSensor(
        task_id="wait_for_predict_api",
        http_conn_id="predict_api",
        endpoint="/health",
        method="GET",
        response_check=lambda response: response.status_code == 200,
        timeout=300, poke_interval=15, mode="reschedule",
    )

    wait_for_minilm_encoder = HttpSensor(
        task_id="wait_for_minilm_encoder",
        http_conn_id="minilm_encoder",
        endpoint="/health",
        method="GET",
        response_check=lambda response: response.status_code == 200,
        timeout=300, poke_interval=15, mode="reschedule",
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

    # ── 1. CV training ────────────────────────────────────────────────────────
    trigger_cv = PythonOperator(
        task_id="trigger_cv_training",
        python_callable=trigger_cv_training,
        provide_context=True,
    )

    wait_for_cv = TrainingCompleteSensor(
        task_id="wait_for_cv_training",
        job_var_key=_CV_JOB_ID_VAR,
        mode="reschedule",
        poke_interval=300,
        timeout=TRAINING_MAX_WAIT,
        soft_fail=False,
    )

    # ── 2. MiniLM encoding (model unloads when done, frees RAM for training) ──
    start_encoding = PythonOperator(
        task_id="trigger_minilm_encoding",
        python_callable=trigger_minilm_encoding,
        provide_context=True,
    )

    wait_for_encoding = MiniLMEncodingSensor(
        task_id="wait_for_minilm_encoding",
        mode="reschedule",
        poke_interval=60,           # poll every minute (encoding takes ~30 min)
        timeout=4 * 3600,
        soft_fail=False,
    )

    # ── 3. MiniLM training ────────────────────────────────────────────────────
    trigger_minilm = PythonOperator(
        task_id="trigger_minilm_training",
        python_callable=trigger_minilm_training,
        provide_context=True,
    )

    wait_for_minilm = TrainingCompleteSensor(
        task_id="wait_for_minilm_training",
        job_var_key=_MINILM_JOB_ID_VAR,
        mode="reschedule",
        poke_interval=300,
        timeout=TRAINING_MAX_WAIT,
        soft_fail=False,
    )

    # ── 4. Post-processing ────────────────────────────────────────────────────
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
            "echo '=== MLOps Pipeline Completed — Both Models Trained ===' && "
            "echo 'CV model    : artifacts/neural_network_model.keras' && "
            "echo 'MiniLM model: artifacts/neural_network_model_minilm.keras' && "
            "echo 'Grafana  : http://localhost:3000' && "
            "echo 'Streamlit: http://localhost:8501' && "
            "echo 'Pipeline executed: {{ ds }}'"
        ),
    )

    # --- Task Dependencies ---
    (
        check_data
        >> [wait_for_gate_api, wait_for_train_api, wait_for_predict_api, wait_for_minilm_encoder]
        >> dataset_stats
        >> get_token
        >> trigger_cv
        >> wait_for_cv
        >> start_encoding
        >> wait_for_encoding
        >> trigger_minilm
        >> wait_for_minilm
        >> [get_version, push_metrics]
        >> eval_results
        >> verify_mlflow
        >> success_message
    )
