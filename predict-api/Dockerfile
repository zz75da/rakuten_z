# predict-api/Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    libjemalloc2 tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng \
    && JEMALLOC=$(find /usr/lib -name 'libjemalloc.so.2' 2>/dev/null | head -1) \
    && [ -n "$JEMALLOC" ] && ln -sf "$JEMALLOC" /usr/local/lib/libjemalloc.so.2 || true \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

# Pre-install CPU-only torch so sentence-transformers doesn't pull CUDA wheels
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# spaCy models for CV/TF-IDF text preprocessing — must match train-api's
# preprocess_text.py (fr_core_news_sm primary, en_core_web_sm fallback)
RUN python -m spacy download fr_core_news_sm && \
    python -m spacy download en_core_web_sm

COPY . .

EXPOSE 5003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]
