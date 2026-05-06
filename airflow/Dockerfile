#airflow/Dockerfile
FROM ghcr.io/YOUR_GITHUB_USERNAME/rakuten-mlops-base:latest

USER root

RUN apt-get update && apt-get install -y postgresql-client jq && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/airflow/logs && chmod -R 777 /tmp/airflow/logs

USER airflow

COPY airflow/requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt \
    apache-airflow-providers-http PyJWT prometheus-client

WORKDIR /opt/airflow

COPY airflow/dags /opt/airflow/dags
COPY airflow/plugins /opt/airflow/plugins
COPY airflow/airflow.cfg /opt/airflow/airflow.cfg
COPY params.yaml /app/params.yaml
COPY --chown=airflow:root airflow/airflow_init.sh /app/airflow_init.sh

RUN chmod +x /app/airflow_init.sh

EXPOSE 8080

CMD bash -c "dvc pull && airflow standalone"
