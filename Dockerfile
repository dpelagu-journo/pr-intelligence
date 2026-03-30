FROM python:3.11-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar el modelo de embeddings en la imagen
# (evita descarga de ~120 MB en cada arranque en producción)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

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
