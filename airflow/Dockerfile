#airflow/Dockerfile
FROM apache/airflow:2.9.1-python3.11

USER root

RUN apt-get update && apt-get install -y postgresql-client jq && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/airflow/logs && chmod -R 777 /tmp/airflow/logs

USER airflow

COPY airflow/requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps \
        apache-airflow-providers-http==4.10.1 \
        apache-airflow-providers-docker==3.12.1 \
    && pip install \
        "dvc[s3]==3.62.0" \
        mlflow==2.8.1 \
        numpy==1.26.4 \
        pandas==2.2.2 \
        "scikit-learn==1.4.2" \
        prometheus-client==0.20.0 \
        PyJWT \
        python-dateutil

WORKDIR /opt/airflow

COPY airflow/dags /opt/airflow/dags
COPY airflow/plugins /opt/airflow/plugins
COPY airflow/airflow.cfg /opt/airflow/airflow.cfg
COPY params.yaml /app/params.yaml
COPY --chown=airflow:root airflow/airflow_init.sh /app/airflow_init.sh

RUN chmod +x /app/airflow_init.sh

EXPOSE 8080

CMD bash -c "dvc pull && airflow standalone"
