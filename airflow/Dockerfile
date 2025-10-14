FROM apache/airflow:2.10.2

USER root

# Install system dependencies + AWS CLI + git for DVC
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    jq \
    git \
    unzip \
    postgresql-client \
    libpq-dev \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws \
    && rm -rf /var/lib/apt/lists/*

# Create log directory with proper permissions to avoid Windows issues
RUN mkdir -p /tmp/airflow/logs && chmod -R 777 /tmp/airflow/logs

USER airflow

# Copy and install requirements
COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Extra Python deps
RUN pip install apache-airflow-providers-http \
    PyJWT \
    mlflow \
    prometheus-client \
    dvc[s3]


RUN pip install --no-cache-dir dagshub

# Create directories for data access
RUN mkdir -p /opt/airflow/data /opt/airflow/artifacts

WORKDIR /opt/airflow

# Copy DAGs and configuration
COPY airflow/dags /opt/airflow/dags
COPY airflow/plugins /opt/airflow/plugins
COPY airflow/airflow.cfg /opt/airflow/airflow.cfg
COPY params.yaml /app/params.yaml

COPY --chown=airflow:root airflow/airflow_init.sh /app/airflow_init.sh
RUN chmod +x /app/airflow_init.sh

EXPOSE 8080

# Default: sync artifacts then start Airflow
CMD bash -c "dvc pull && airflow standalone"