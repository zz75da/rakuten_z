FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl unzip build-essential libpq-dev \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o aws.zip \
    && unzip aws.zip && ./aws/install \
    && rm -rf aws aws.zip /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel setuptools

RUN pip install \
    dvc[s3]==3.62.0 \
    dagshub==0.6.3 \
    mlflow==2.8.0 \
    boto3==1.34.162 \
    psycopg2-binary==2.9.9
