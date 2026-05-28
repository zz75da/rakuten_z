# CPU-only TF: no BFCAllocator/CUDA memory arena, no glibc heap corruption during model.fit()
FROM tensorflow/tensorflow:2.17.0

WORKDIR /app

# 1. Installation des dépendances système (on le fait en premier pour le cache)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    libjemalloc2 \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
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

# 5. Téléchargement des modèles Spacy
# fr_core_news_sm: correct French lemmatization + French stopwords (49% of Rakuten data is French)
# en_core_web_sm:  kept as fallback for English product titles
RUN python -m spacy download fr_core_news_sm && \
    python -m spacy download en_core_web_sm

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002"]