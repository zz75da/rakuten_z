
# Control Check Memo – Airflow + Postgres + Train API + MLflow/DagsHub

This memo contains ready-to-run commands for debugging your setup when using Docker Compose.  
Each section includes **success vs. error expectations** so you can quickly identify issues.

---

## 1. Airflow → Postgres Connection Check

### Command (inside Airflow container):
```bash
docker compose exec airflow-scheduler ping -c 2 postgres
docker compose exec airflow-scheduler psql -h postgres -U <username> -d <dbname> -c "SELECT 1;"
```

### Success:
- `ping` shows replies like `64 bytes from postgres...`
- `psql` prints a table with `?column? | 1`

### Error:
- `ping: bad address 'postgres'` → hostname not resolvable
- `could not translate host name` → DNS issue
- `psql: could not connect` → database down or wrong credentials

---

## 2. Airflow → Train API Connectivity Check

### Command:
```bash
docker compose exec airflow-scheduler curl -v http://train-api:5002/health
docker compose exec airflow-scheduler curl -v http://train-api:5002/train
```

### Success:
- `/health` → returns `{"status":"ok"}` or HTTP 200
- `/train` → starts training (may take time)

### Error:
- `Connection refused` → container not running
- `Could not resolve host: train-api` → networking issue
- `500 Internal Server Error` → train-api crashed (check its logs)

---

## 3. MLflow Environment Variables

### Command:
```bash
docker compose exec train-api bash -c 'echo $MLFLOW_TRACKING_URI && echo $MLFLOW_S3_ENDPOINT_URL && echo $MLFLOW_ARTIFACT_URI && echo $AWS_ACCESS_KEY_ID && echo $AWS_SECRET_ACCESS_KEY && echo $AWS_DEFAULT_REGION'
```

### Success:
- Shows your MLflow + AWS/DagsHub configs (`http://mlflow:5000`, `s3://<dagshub-bucket>` etc.)

### Error:
- Blank values → environment variables not passed
- Typos or spaces (e.g., `' us-east-1'`) → MLflow will break with region errors

---

## 4. Verify MLflow Artifacts in DagsHub S3

### Command:
```bash
docker compose exec train-api aws s3 ls s3://<your-dagshub-bucket>/ --endpoint-url https://dagshub.com
```

### Success:
- Lists files like `model.keras`, `metrics.json`, `plots/history.json`

### Error:
- `AccessDenied` → wrong AWS credentials
- `Could not connect to the endpoint URL` → bad endpoint config
- Empty result but training ran → MLflow only logged runs locally, not to S3

---

## 5. Airflow Scheduler Health

### Command:
```bash
docker compose logs airflow-scheduler | grep heartbeat
```

### Success:
- Repeated logs like: `Scheduler heartbeat`

### Error:
- `Job heartbeat failed` or `scheduler in unhealthy state` → usually Postgres connection issue

---

## 6. Bonus: Check All Running Containers

```bash
docker compose ps
```

- **Success**: `airflow-scheduler`, `train-api`, `mlflow`, `postgres` all in `Up` state
- **Error**: `Exit` or `Restarting` → check logs

---

✅ With these checks, you’ll know exactly where the pipeline is failing.
