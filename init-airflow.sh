FROM apache/airflow:2.8.1

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install apache-airflow-providers-http mlflow prometheus-client PyJWT

COPY airflow/dags /opt/airflow/dags
COPY airflow/plugins /opt/airflow/plugins
COPY airflow/airflow.cfg /opt/airflow/airflow.cfg
COPY params.yaml /app/params.yaml
COPY scripts/init-airflow.sh /scripts/init-airflow.sh

USER airflow