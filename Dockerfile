FROM python:3.11-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ruta explícita del caché de fastembed (consistente entre build y runtime)
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed

# Pre-descargar el modelo ONNX de embeddings en la imagen (~60 MB, sin PyTorch)
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').embed(['test']))"

# Copiar código
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Directorios de datos (Railway los mapeará a un volumen)
RUN mkdir -p /data/vector_store /uploads

# Variables de entorno por defecto
ENV VECTOR_STORE_PATH=/data/vector_store
ENV UPLOAD_DIR=/uploads
ENV PYTHONUNBUFFERED=1

WORKDIR /app/backend

EXPOSE 8000

# Railway inyecta $PORT automáticamente
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
