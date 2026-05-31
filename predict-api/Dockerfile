# predict-api/Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    libjemalloc2 \
    && JEMALLOC=$(find /usr/lib -name 'libjemalloc.so.2' 2>/dev/null | head -1) \
    && [ -n "$JEMALLOC" ] && ln -sf "$JEMALLOC" /usr/local/lib/libjemalloc.so.2 || true \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

# Pre-install CPU-only torch so sentence-transformers doesn't pull CUDA wheels
# (nvidia 2.7 GB + triton 700 MB) from the PyPI default index.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Keras 3 / TF 2.17 compatibility — patch load_context.py wherever it is installed
RUN find /usr/local/lib -name "load_context.py" -path "*/saved_model/*" \
    -exec sed -i 's/tf\.__internal__\.register_load_context_function(in_load_context)/pass  # patched: removed in TF 2.17/' {} \; \
    -exec echo "Patched: " {} \; 2>/dev/null || true

COPY . .

EXPOSE 5003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]
