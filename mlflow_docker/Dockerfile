# mlflow_docker/Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    curl \
    libyaml-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY mlflow_docker/requirements.txt .

# Install pip first (to get newer version that handles wheels better)
RUN pip install --no-cache-dir --upgrade pip

# Install requirements (let pip choose compatible PyYAML wheel)
RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt

# Create directory for MLflow
RUN mkdir -p /mlflow

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]