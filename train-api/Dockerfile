# train-api  (tensorflow/tensorflow:2.17.0-gpu bundles CUDA 12 + cuDNN — no host install needed)
FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app

# 1. Installation des dépendances système (on le fait en premier pour le cache)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Copie EXPLICITE du requirements
COPY requirements.txt /app/requirements.txt

# 3. Installation avec chemin ABSOLU vers le fichier
# Note : Vérifie que httpx est bien écrit dans ton fichier requirements.txt local
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --ignore-installed blinker && \
    pip install -r /app/requirements.txt

# 4. Copie du reste du code
COPY . .

# 5. Téléchargement du modèle Spacy
RUN python -m spacy download en_core_web_sm

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002"]