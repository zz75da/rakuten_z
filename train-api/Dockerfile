# train-api
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt dvc[s3]

# Install system deps + AWS CLI
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        unzip \
        git \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY . .

ENV GATE_API_URL=http://gate-api:5000
ENV SECRET_KEY=default_secret

RUN python -m spacy download en_core_web_sm

EXPOSE 5002


