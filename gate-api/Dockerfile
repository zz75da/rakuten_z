# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps including DVC
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt dvc[s3]

# Copy app code
COPY app.py .

# Ensure directories exist
RUN mkdir -p /app/data/feature_cache /app/data/images /app/data/artifacts

EXPOSE 5000

CMD ["bash", "-c", "dvc pull || true && uvicorn app:app --host 0.0.0.0 --port 5000"]
#CMD bash -c "dvc pull && uvicorn app:app --host 0.0.0.0 --port 5000"