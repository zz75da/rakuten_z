from datetime import datetime, timedelta, timezone
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.exceptions import AirflowException
from airflow.utils.trigger_rule import TriggerRule
import requests
import os
from pathlib import Path

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
CLIP_ENCODER     = "http://clip-encoder:5007"

TRAINING_MAX_WAIT      = int(os.environ.get("TRAINING_MAX_WAIT_SECONDS",  30 * 3600))
ENCODING_MAX_WAIT      = int(os.environ.get("ENCODING_MAX_WAIT_SECONDS",  16 * 3600))

# Airflow Variable keys — one per training job so they don't collide
_CV_JOB_ID_VAR     = "rakuten_cv_training_job_id"
_CLIP_JOB_ID_VAR   = "rakuten_clip_training_job_id"
_MINILM_JOB_ID_VAR = "rakuten_minilm_training_job_id"
_MPNET_JOB_ID_VAR  = "rakuten_mpnet_training_job_id"

# Quality gate — fail before wasting 10 h of training on a corrupted CSV
_MIN_DATASET_ROWS = 80_000

# Regression gate — stored best-ever val_acc per encoder
_CV_BEST_VAL_ACC_VAR     = "rakuten_cv_best_val_acc"
_CLIP_BEST_VAL_ACC_VAR   = "rakuten_clip_best_val_acc"
_MINILM_BEST_VAL_ACC_VAR = "rakuten_minilm_best_val_acc"
_MPNET_BEST_VAL_ACC_VAR  = "rakuten_mpnet_best_val_acc"
_REGRESSION_WARN_PCT     = 2.0   # print warning if either model drops > 2 %
_REGRESSION_FAIL_PCT     = 5.0   # block DVC push only if BOTH models drop > 5 % simultaneously


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
    import pandas as pd
    csv_path = "/opt/airflow/data/X_train_update.csv"
    full = pd.read_csv(csv_path)
    n_rows, n_cols = len(full), len(full.columns)
    shape = f"{n_rows}x{n_cols}"
    context["ti"].xcom_push(key="data_shape", value=shape)
    context["ti"].xcom_push(key="data_columns", value=list(full.columns))
    print(f"Dataset: {shape}, columns: {list(full.columns)}")
    if n_rows < _MIN_DATASET_ROWS:
        raise AirflowException(
            f"Dataset too small: {n_rows} rows (threshold={_MIN_DATASET_ROWS}). "
            "Check /opt/airflow/data/X_train_update.csv for corruption or truncation."
        )
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
            elif probe.status_code == 200:
                probe_status = probe.json().get("status")
                if probe_status in ("interrupted", "failed"):
                    print(f"Job {existing_job_id} status={probe_status} — starting fresh.")
                    Variable.delete(job_var_key)
                    existing_job_id = None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            print(f"Probe timed out for {existing_job_id} — assuming still running.")
        if existing_job_id:
            print(f"Resuming existing {text_encoder} job: {existing_job_id}")
            context["ti"].xcom_push(key=f"{text_encoder}_job_id", value=existing_job_id)
            return existing_job_id

    def _read_epochs():
        for path in ["/opt/airflow/params.yaml", "/app/params.yaml"]:
            if os.path.exists(path):
                import yaml
                p = yaml.safe_load(open(path)) or {}
                return int(p.get("train", {}).get("epochs", 30))
        return int(os.environ.get("TRAIN_EPOCHS", 30))

    def _git_sha():
        # git is not installed in the airflow container; read .git directly
        try:
            git_dir = "/opt/airflow/.git"
            head = open(f"{git_dir}/HEAD").read().strip()
            if head.startswith("ref: "):
                ref_path = f"{git_dir}/{head[5:]}"
                return open(ref_path).read().strip()
            return head  # detached HEAD — already a SHA
        except Exception:
            return ""

    payload = {
        "use_dev_images": False,
        "epochs": _read_epochs(),
        "batch_size": int(os.environ.get("TRAIN_BATCH_SIZE", 128)),
        "text_encoder": text_encoder,
        "git_commit_sha": _git_sha(),
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


def trigger_clip_encoding(**context):
    """Call POST /encode on the clip-encoder service."""
    resp = requests.post(f"{CLIP_ENCODER}/encode", timeout=30)
    data = resp.json()
    print(f"CLIP encoder: {data}")
    if data.get("status") == "error":
        raise AirflowException(f"CLIP encoding error: {data.get('message')}")


def trigger_clip_training(**context):
    return _trigger_training_job(_CLIP_JOB_ID_VAR, "clip", context)


def trigger_minilm_encoding(**context):
    """Call POST /encode on the minilm-encoder service."""
    resp = requests.post(f"{MINILM_ENCODER}/encode", timeout=30)
    data = resp.json()
    print(f"MiniLM encoder: {data}")
    if data.get("status") == "error":
        raise AirflowException(f"MiniLM encoding error: {data.get('message')}")


def trigger_minilm_training(**context):
    return _trigger_training_job(_MINILM_JOB_ID_VAR, "minilm", context)


def trigger_mpnet_training(**context):
    return _trigger_training_job(_MPNET_JOB_ID_VAR, "mpnet", context)


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

        try:
            token = _fresh_token()
        except Exception as exc:
            print(f"Login to gate-api failed (will retry in 5 min): {exc}")
            return False
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

        if resp.status_code in (503, 502, 504):
            print(f"train-api returned {resp.status_code} (overloaded) — will retry in 5 min")
            return False

        if resp.status_code == 404:
            Variable.delete(self.job_var_key)
            raise AirflowException(
                f"Job {job_id} not found (train-api restarted). "
                "Clear the DAG run and re-trigger to start a new job."
            )
        resp.raise_for_status()

        job    = resp.json()
        status = job.get("status")
        step   = job.get("step", "")

        # Build a detailed log line for every poke
        log_parts = [f"Job {job_id} [{self.job_var_key}]: status={status}"]
        if step:
            log_parts.append(f"step={step}")
        started_at = job.get("started_at")
        if started_at:
            try:
                from dateutil import parser as _dtparser
                elapsed_s = (datetime.now(timezone.utc) - _dtparser.parse(started_at)).total_seconds()
                log_parts.append(f"elapsed={elapsed_s / 60:.1f}min")
            except Exception:
                pass
        if job.get("dataset_size"):
            log_parts.append(f"dataset_size={job['dataset_size']}")
        if job.get("num_classes"):
            log_parts.append(f"num_classes={job['num_classes']}")
        print(" | ".join(log_parts))

        if status == "success":
            Variable.delete(self.job_var_key)
            final = job.get("final_metrics", {})
            history = job.get("history", {})
            val_accs = history.get("val_accuracy", [])
            print(
                f"  ✓ Training complete — "
                f"accuracy={final.get('accuracy', 'N/A')} | "
                f"best_val_acc={max(val_accs):.4f} | "
                f"epochs={len(history.get('loss', []))} | "
                f"mlflow_run={job.get('mlflow_run_id', 'N/A')}"
            )
            context["ti"].xcom_push(key="training_result", value=job)
            return True

        if status == "failed":
            Variable.delete(self.job_var_key)
            raise AirflowException(
                f"Training job {job_id} failed at step={step!r}: {job.get('error', 'unknown error')}"
            )

        if status == "interrupted":
            Variable.delete(self.job_var_key)
            raise AirflowException(
                f"Job {job_id} was interrupted (train-api restarted mid-training). "
                "Clear this DAG run and re-trigger."
            )

        # status == "running" → reschedule
        return False


class ClipEncodingSensor(BaseSensorOperator):
    """Polls GET /status on the clip-encoder service until encoding is done."""

    def poke(self, context):
        try:
            resp   = requests.get(f"{CLIP_ENCODER}/status", timeout=30)
            data   = resp.json()
            status = data.get("status")
            print(f"CLIP encoder status={status}: {data.get('message', '')}")
            if status == "done":
                return True
            if status == "error":
                raise AirflowException(f"CLIP encoding failed: {data.get('message')}")
            return False
        except AirflowException:
            raise
        except Exception as exc:
            print(f"CLIP status check failed (retrying): {exc}")
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
    """Push training metrics for CV, CLIP, MiniLM, and mpnet to Prometheus Pushgateway."""
    from dateutil import parser as dtparser

    runs = [
        ("wait_for_cv_training",    "countvectorizer"),
        ("wait_for_clip_training",  "clip"),
        ("wait_for_minilm_training", "minilm"),
        ("wait_for_mpnet_training",  "mpnet"),
    ]
    for task_id, encoder in runs:
        result  = context["ti"].xcom_pull(task_ids=task_id, key="training_result") or {}
        final   = result.get("final_metrics", {})
        history = result.get("history", {})
        if not history:
            print(f"No history for {encoder} — skipping push")
            continue

        val_accuracy = (history.get("val_accuracy") or [0])[-1]
        loss         = (history.get("loss") or [0])[-1]
        val_loss     = (history.get("val_loss") or [0])[-1]
        epochs_done  = len(history.get("loss") or [])

        started_at   = result.get("started_at", "")
        completed_at = result.get("completed_at", "")
        elapsed = 0
        if started_at and completed_at:
            try:
                elapsed = (dtparser.parse(completed_at) - dtparser.parse(started_at)).total_seconds()
            except Exception:
                pass

        # Use encoder label in the Pushgateway job path so both encoders
        # can coexist without overwriting each other's metrics.
        metrics = {
            "training_accuracy":     final.get("accuracy", 0),
            "validation_accuracy":   val_accuracy,
            "training_loss":         loss,
            "validation_loss":       val_loss,
            "epochs_completed":      epochs_done,
            "training_time_seconds": elapsed,
        }
        metrics_text = "\n".join(f"{k} {v}" for k, v in metrics.items())
        try:
            resp = requests.post(
                f"http://pushgateway:9091/metrics/job/rakuten_mlops/encoder/{encoder}",
                data=metrics_text, timeout=10,
            )
            print(f"{encoder} metrics pushed (HTTP {resp.status_code}): {metrics}")
        except Exception as exc:
            print(f"Prometheus push failed for {encoder} (non-blocking): {exc}")


def push_feature_cache(**context):
    """
    Commit and push all DVC-tracked feature cache files to both DagsHub remotes.
    Runs DVC directly in the Airflow container (project root at /opt/airflow).
    Skipped when the DAG is triggered with conf={'push_dvc_cache': false}.

    Workflow:
      1. dvc commit -f  — tells DVC about outputs produced by Docker (outside dvc repro).
                          This updates dvc.lock with current file hashes without needing
                          dvc add, avoiding conflicts with existing stage declarations.
      2. dvc push       — uploads all committed outputs to both remotes (non-fatal).

    Both remotes point to DagsHub (S3 + HTTP). Network timeouts are non-fatal so
    Prometheus metrics and model evaluation still run even if DagsHub is unreachable.
    """
    import subprocess

    push_enabled = True
    if context.get("dag_run") and context["dag_run"].conf:
        push_enabled = context["dag_run"].conf.get("push_dvc_cache", True)
    if not push_enabled:
        print("push_dvc_cache=false — skipping DVC cache push")
        return {"skipped": True}

    dvc_root = "/opt/airflow"

    # Configure dagshub HTTP remote credentials from env (config.local is gitignored)
    dagshub_user  = os.getenv("DAGSHUB_USER", "")
    dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
    if dagshub_user and dagshub_token:
        for key, val in [("auth", "basic"), ("user", dagshub_user), ("password", dagshub_token)]:
            subprocess.run(
                ["dvc", "remote", "modify", "dagshub", "--local", key, val],
                capture_output=True, cwd=dvc_root,
            )
        print(f"dagshub HTTP remote credentials configured for user={dagshub_user}")

    # dvc commit -f: register Docker-produced outputs with DVC without running dvc repro.
    # This resolves the "output specified in stage AND .dvc file" conflict by updating
    # dvc.lock rather than creating separate .dvc pointer files via dvc add.
    commit = subprocess.run(
        ["dvc", "commit", "-f"],
        capture_output=True, text=True, cwd=dvc_root,
    )
    print(f"dvc commit -f: rc={commit.returncode} | {commit.stdout.strip()} | {commit.stderr.strip()[:300]}")
    if commit.returncode != 0:
        print("Warning: dvc commit failed (non-fatal) — will still attempt push")

    results = {}
    any_ok = False
    for remote in ["origin", "dagshub"]:
        push = subprocess.run(
            ["dvc", "push", "--remote", remote],
            capture_output=True, text=True, cwd=dvc_root,
        )
        print(f"dvc push --remote {remote}: rc={push.returncode} | {push.stdout.strip()} | {push.stderr.strip()[:300]}")
        if push.returncode == 0:
            any_ok = True
        else:
            print(f"Warning: dvc push to {remote} failed (non-fatal) — pipeline continues")
        results[remote] = {"rc": push.returncode, "output": push.stdout.strip()}

    if not any_ok:
        print("Warning: both DVC remotes failed — feature cache not pushed this run")

    # Commit updated dvc.lock to git so a fresh clone can reproduce the run.
    # Non-fatal: if git push fails (no credentials, detached HEAD, etc.) the
    # binary files are already in DagsHub S3; only reproducibility tracking is
    # affected.
    git_ok = False
    try:
        # Configure git identity (required in the airflow container)
        subprocess.run(["git", "config", "user.email", "airflow@rakuten-mlops"], cwd=dvc_root, capture_output=True)
        subprocess.run(["git", "config", "user.name",  "Airflow DAG"],           cwd=dvc_root, capture_output=True)

        add = subprocess.run(
            ["git", "add", "dvc.lock"],
            capture_output=True, text=True, cwd=dvc_root,
        )
        print(f"git add dvc.lock: rc={add.returncode} | {add.stderr.strip()[:200]}")

        status = subprocess.run(
            ["git", "status", "--short", "dvc.lock"],
            capture_output=True, text=True, cwd=dvc_root,
        )
        if not status.stdout.strip():
            print("dvc.lock unchanged — no git commit needed")
            git_ok = True
        else:
            # Embed today's date so the commit message is always unique
            from datetime import date as _date
            msg = f"dvc: update lock after pipeline run {_date.today().isoformat()}"
            commit_git = subprocess.run(
                ["git", "commit", "-m", msg],
                capture_output=True, text=True, cwd=dvc_root,
            )
            print(f"git commit: rc={commit_git.returncode} | {commit_git.stdout.strip()} | {commit_git.stderr.strip()[:200]}")

            # Try both remotes; succeed if either works
            for git_remote in ["origin", "dagshub"]:
                push_git = subprocess.run(
                    ["git", "push", git_remote, "HEAD"],
                    capture_output=True, text=True, cwd=dvc_root,
                )
                print(f"git push {git_remote} HEAD: rc={push_git.returncode} | {push_git.stderr.strip()[:200]}")
                if push_git.returncode == 0:
                    git_ok = True
                    break
            if not git_ok:
                print("Warning: git push failed on both remotes — dvc.lock not updated in git (binary files still in S3)")
    except Exception as exc:
        print(f"Warning: git commit/push of dvc.lock failed (non-fatal): {exc}")

    return {"status": "ok", "push_results": results, "dvc_lock_git_pushed": git_ok}


def evaluate_from_result(**context):
    """Log evaluation results for CV, CLIP, MiniLM, and mpnet training runs."""
    for task_id, label in [
        ("wait_for_cv_training",    "CountVectorizer"),
        ("wait_for_clip_training",  "CLIP ViT-B/32"),
        ("wait_for_minilm_training", "MiniLM-L12"),
        ("wait_for_mpnet_training",  "mpnet-base-v2"),
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


def run_quality_gate(**context):
    """
    POST /quality-gate on train-api — runs pytest model quality assertions.
    Raises AirflowException on failure so the DAG blocks deployment.
    """
    token = context["ti"].xcom_pull(task_ids="get_auth_token", key="auth_token")
    if not token:
        raise AirflowException("No auth token — cannot run quality gate")
    resp = requests.post(
        f"{TRAIN_API}/quality-gate",
        headers={"Authorization": f"Bearer {token}"},
        timeout=180,
    )
    output = ""
    try:
        output = resp.json().get("output", "")
    except Exception:
        output = resp.text[:2000]
    print(output)
    if not resp.ok:
        raise AirflowException(f"Quality gate FAILED (HTTP {resp.status_code})\n{output[-1000:]}")
    print("Quality gate PASSED")
    return {}


def rebuild_drift_reference(**context):
    """
    POST /drift-rebuild-reference on train-api.
    Builds the 5k stratified reference CSV for Evidently drift monitoring.
    Non-fatal — drift reference failure must not block deployment.
    """
    token = context["ti"].xcom_pull(task_ids="get_auth_token", key="auth_token")
    if not token:
        print("No auth token — skipping drift reference rebuild")
        return {}
    try:
        resp = requests.post(
            f"{TRAIN_API}/drift-rebuild-reference",
            params={"n_samples": 5000},
            headers={"Authorization": f"Bearer {token}"},
            timeout=120,
        )
        if resp.ok:
            d = resp.json()
            print(f"Drift reference built: {d.get('rows')} rows → {d.get('path')}")
        else:
            print(f"Drift reference returned {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"Drift reference rebuild failed (non-fatal): {e}")
    return {}


def run_cleanlab_audit(**context):
    """
    POST /cleanlab on train-api using the CLIP model (best accuracy).
    Non-fatal — a label audit failure must not block deployment.
    Timeout: 15 min (cleanlab on 85k samples takes ~5-10 min).
    """
    token = context["ti"].xcom_pull(task_ids="get_auth_token", key="auth_token")
    if not token:
        print("No auth token available — skipping cleanlab audit")
        return {}
    try:
        resp = requests.post(
            f"{TRAIN_API}/cleanlab",
            params={"encoder": "clip"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=960,   # 16 min — slightly above the 15-min server cap
        )
        if resp.ok:
            data = resp.json()
            print(f"Cleanlab audit complete — {data.get('n_issues')} issues found")
            print(f"Report: {data.get('report')}")
        else:
            print(f"Cleanlab audit returned {resp.status_code}: {resp.text[:500]}")
    except Exception as e:
        print(f"Cleanlab audit failed (non-fatal): {e}")
    return {}


_RUN_LOG = Path("/opt/airflow/data/dag_runs.log")
_RUN_SEP = "=" * 72
_MAX_RUNS = 3


def write_run_summary(**context):
    """
    Always runs (TriggerRule.ALL_DONE) — writes a compact run summary to
    /opt/airflow/data/dag_runs.log, keeping only the last 3 complete runs.
    This gives a quick human-readable audit trail without opening the Airflow UI.
    """
    dag_run = context["dag_run"]
    ti      = context["ti"]

    cv_result     = ti.xcom_pull(task_ids="wait_for_cv_training",    key="training_result") or {}
    clip_result   = ti.xcom_pull(task_ids="wait_for_clip_training",  key="training_result") or {}
    ml_result     = ti.xcom_pull(task_ids="wait_for_minilm_training", key="training_result") or {}
    mpnet_result  = ti.xcom_pull(task_ids="wait_for_mpnet_training",  key="training_result") or {}
    model_version = ti.xcom_pull(task_ids="get_model_version",       key="model_version") or "N/A"

    def _model_lines(label, result):
        h = result.get("history", {})
        val_accs = h.get("val_accuracy", [])
        return [
            f"  {label}:",
            f"    best_val_acc : {max(val_accs):.4f}" if val_accs else "    best_val_acc : N/A",
            f"    epochs       : {len(h.get('loss', []))}",
            f"    mlflow_run   : {result.get('mlflow_run_id', 'N/A')}",
        ]

    block_lines = [
        f"RUN  : {dag_run.run_id}",
        f"State: {dag_run.state}",
        f"Start: {dag_run.start_date}",
        f"End  : {dag_run.end_date}",
        f"MReg : {model_version}",
        "",
    ] + _model_lines("CV  (countvectorizer)", cv_result) \
      + [""] + _model_lines("CLIP ViT-B/32", clip_result) \
      + [""] + _model_lines("MiniLM-L12", ml_result) \
      + [""] + _model_lines("mpnet-base-v2", mpnet_result)

    block = "\n".join(block_lines)

    _RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    existing = _RUN_LOG.read_text() if _RUN_LOG.exists() else ""
    blocks = [b for b in existing.split(_RUN_SEP) if b.strip()]
    blocks.append(block)
    blocks = blocks[-_MAX_RUNS:]
    _RUN_LOG.write_text(f"\n{_RUN_SEP}\n".join(blocks))
    print(f"Run summary written → {_RUN_LOG}  (last {len(blocks)} runs kept)")


def check_regression_gate(**context):
    """
    Runs after both training sensors complete, before DVC cache push.

    - Warns if either model's best val_acc drops > _REGRESSION_WARN_PCT vs stored best-ever.
    - Blocks DVC push (raises AirflowException) only if BOTH models simultaneously drop
      > _REGRESSION_FAIL_PCT — a catastrophic run, not a single-encoder wobble.
    - On first run (no baseline stored) just saves current values and passes.
    - Always updates best-ever to max(current, stored) so the bar never lowers.
    """
    ti = context["ti"]

    encoder_cfg = [
        ("countvectorizer", "wait_for_cv_training",    _CV_BEST_VAL_ACC_VAR),
        ("clip",            "wait_for_clip_training",   _CLIP_BEST_VAL_ACC_VAR),
        ("minilm",          "wait_for_minilm_training", _MINILM_BEST_VAL_ACC_VAR),
        ("mpnet",           "wait_for_mpnet_training",  _MPNET_BEST_VAL_ACC_VAR),
    ]

    regressions = {}

    for encoder, task_id, var_key in encoder_cfg:
        result   = ti.xcom_pull(task_ids=task_id, key="training_result") or {}
        history  = result.get("history", {})
        val_accs = history.get("val_accuracy", [])
        if not val_accs:
            print(f"{encoder}: no val_accuracy in training_result — skipping regression check")
            continue

        current_best = max(val_accs)
        stored_str   = Variable.get(var_key, default_var=None)

        if stored_str is None:
            Variable.set(var_key, str(current_best))
            print(f"{encoder}: first run — baseline stored → {current_best:.4f}")
            continue

        stored_best = float(stored_str)
        drop_pct = (stored_best - current_best) / stored_best * 100 if stored_best > 0 else 0.0

        if drop_pct > _REGRESSION_WARN_PCT:
            print(
                f"WARNING {encoder}: val_acc dropped {drop_pct:.1f}% "
                f"(best-ever={stored_best:.4f}, this run={current_best:.4f})"
            )
            regressions[encoder] = drop_pct
        else:
            print(
                f"{encoder}: val_acc OK — best-ever={stored_best:.4f}, "
                f"this run={current_best:.4f} (Δ{-drop_pct:+.1f}%)"
            )

        new_best = max(current_best, stored_best)
        Variable.set(var_key, str(new_best))
        print(f"{encoder}: best-ever updated → {new_best:.4f}")

    all_regressed = (
        len(regressions) >= 2
        and all(v > _REGRESSION_FAIL_PCT for v in regressions.values())
    )
    if all_regressed:
        details = ", ".join(f"{enc}: -{pct:.1f}%" for enc, pct in regressions.items())
        raise AirflowException(
            f"All regressed encoders dropped > {_REGRESSION_FAIL_PCT}% — blocking DVC cache push. "
            f"{details}. Review hyperparameters and re-run the DAG."
        )


def _dag_failure_callback(context):
    """DAG-level callback: logs which task failed and its error to the run log."""
    failed_ti = context.get("task_instance")
    dag_run   = context.get("dag_run")
    exception = context.get("exception", "unknown")

    msg = (
        f"DAG FAILED  run_id={dag_run.run_id if dag_run else '?'}  "
        f"task={failed_ti.task_id if failed_ti else '?'}  "
        f"error={str(exception)[:300]}"
    )
    print(msg)

    # Append one-liner to the run log so failures are visible without the UI
    _RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _RUN_LOG.open("a") as fh:
        fh.write(f"\n[FAIL] {msg}\n")


# --- DAG Definition ---
with DAG(
    dag_id="rakuten_multimodal_pipeline_v7",
    default_args=default_args,
    description="4-encoder pipeline: late fusion, focal loss, cleanlab audit, quality gate, drift monitoring",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training", "production", "monitoring"],
    on_failure_callback=_dag_failure_callback,
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

    wait_for_clip_encoder = HttpSensor(
        task_id="wait_for_clip_encoder",
        http_conn_id="clip_encoder",
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

    # ── 2. CLIP encoding + training ───────────────────────────────────────────
    start_clip_encoding = PythonOperator(
        task_id="trigger_clip_encoding",
        python_callable=trigger_clip_encoding,
        provide_context=True,
    )

    wait_for_clip_encoding = ClipEncodingSensor(
        task_id="wait_for_clip_encoding",
        mode="reschedule",
        poke_interval=300,
        timeout=ENCODING_MAX_WAIT,
        soft_fail=False,
    )

    trigger_clip = PythonOperator(
        task_id="trigger_clip_training",
        python_callable=trigger_clip_training,
        provide_context=True,
    )

    wait_for_clip = TrainingCompleteSensor(
        task_id="wait_for_clip_training",
        job_var_key=_CLIP_JOB_ID_VAR,
        mode="reschedule",
        poke_interval=300,
        timeout=TRAINING_MAX_WAIT,
        soft_fail=False,
    )

    # ── 3. MiniLM encoding (model unloads when done, frees RAM for training) ──
    start_encoding = PythonOperator(
        task_id="trigger_minilm_encoding",
        python_callable=trigger_minilm_encoding,
        provide_context=True,
    )

    wait_for_encoding = MiniLMEncodingSensor(
        task_id="wait_for_minilm_encoding",
        mode="reschedule",
        poke_interval=300,
        timeout=ENCODING_MAX_WAIT,
        soft_fail=False,
    )

    # ── 4a. MiniLM training (sequential before mpnet — train-api accepts one job at a time) ──
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

    # ── 4b. mpnet training (sequential after MiniLM — avoids 409 + memory pressure) ──
    trigger_mpnet = PythonOperator(
        task_id="trigger_mpnet_training",
        python_callable=trigger_mpnet_training,
        provide_context=True,
    )

    wait_for_mpnet = TrainingCompleteSensor(
        task_id="wait_for_mpnet_training",
        job_var_key=_MPNET_JOB_ID_VAR,
        mode="reschedule",
        poke_interval=300,
        timeout=TRAINING_MAX_WAIT,
        soft_fail=False,
    )

    # ── 5. Regression gate — warn/block before persisting a degraded cache ──────
    regression_gate = PythonOperator(
        task_id="check_regression",
        python_callable=check_regression_gate,
        provide_context=True,
    )

    # ── 6. DVC cache push (optional — set push_dvc_cache=false in DAG run conf to skip) ──
    push_cache = PythonOperator(
        task_id="push_feature_cache",
        python_callable=push_feature_cache,
        provide_context=True,
    )

    # ── 7. Post-processing ────────────────────────────────────────────────────
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
            "BASE=rakuten_multimodal && "
            "if [ -z \"$DAGSHUB_USER\" ]; then echo 'DAGSHUB_USER not set — skipping'; exit 0; fi && "
            "FAILED=0 && "
            "for VARIANT in _cv _clip _minilm _mpnet; do "
            "  MODEL=\"${BASE}${VARIANT}\" && "
            "  URL=\"https://dagshub.com/${DAGSHUB_USER}/rakuten_z.mlflow/api/2.0/mlflow/registered-models/get?name=${MODEL}\" && "
            "  response=$(curl -s -u \"${DAGSHUB_USER}:${DAGSHUB_TOKEN}\" \"$URL\") && "
            "  if echo \"$response\" | grep -q '\"name\"'; then "
            "    echo \"✓ ${MODEL} registered\"; "
            "  else "
            "    echo \"✗ ${MODEL} not found: ${response:0:200}\"; FAILED=1; "
            "  fi; "
            "done && "
            "exit $FAILED"
        ),
    )

    success_message = BashOperator(
        task_id="success_message",
        bash_command=(
            "echo '=== MLOps Pipeline Completed — All 4 Models Trained ===' && "
            "echo 'CV model    : artifacts/neural_network_model.keras' && "
            "echo 'CLIP model  : artifacts/neural_network_model_clip.keras' && "
            "echo 'MiniLM model: artifacts/neural_network_model_minilm.keras' && "
            "echo 'mpnet model : artifacts/neural_network_model_mpnet.keras' && "
            "echo 'Grafana  : http://localhost:3000' && "
            "echo 'Streamlit: http://localhost:8501' && "
            "echo 'Run log  : /opt/airflow/data/dag_runs.log  (last 3 runs)' && "
            "echo 'Pipeline executed: {{ ds }}'"
        ),
    )

    # Always runs — writes last-3-runs summary to /opt/airflow/data/dag_runs.log
    drift_reference = PythonOperator(
        task_id="rebuild_drift_reference",
        python_callable=rebuild_drift_reference,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
        execution_timeout=timedelta(minutes=3),
    )

    quality_gate = PythonOperator(
        task_id="quality_gate",
        python_callable=run_quality_gate,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
    )

    cleanlab_audit = PythonOperator(
        task_id="cleanlab_audit",
        python_callable=run_cleanlab_audit,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,   # only run if all training succeeded
        execution_timeout=timedelta(minutes=18), # hard DAG-level cap above the 15-min server cap
    )

    run_summary = PythonOperator(
        task_id="write_run_summary",
        python_callable=write_run_summary,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,  # runs on success AND failure
    )

    # --- Task Dependencies ---
    # Linear chain up to encoding (CV → CLIP encode+train → MiniLM/mpnet encode)
    (
        check_data
        >> [wait_for_gate_api, wait_for_train_api, wait_for_predict_api,
            wait_for_minilm_encoder, wait_for_clip_encoder]
        >> dataset_stats
        >> get_token
        >> trigger_cv
        >> wait_for_cv
        >> start_clip_encoding
        >> wait_for_clip_encoding
        >> trigger_clip
        >> wait_for_clip
        >> start_encoding
        >> wait_for_encoding
    )

    # Sequential: MiniLM first, then mpnet (train-api only runs one subprocess at a time;
    # parallel would cause the second POST /train to get 409, and doubles memory pressure)
    wait_for_encoding >> trigger_minilm >> wait_for_minilm >> trigger_mpnet >> wait_for_mpnet

    (
        wait_for_mpnet
        >> regression_gate
        >> push_cache
        >> [get_version, push_metrics]
        >> eval_results
        >> verify_mlflow
        >> drift_reference
        >> quality_gate
        >> cleanlab_audit
        >> success_message
        >> run_summary
    )
