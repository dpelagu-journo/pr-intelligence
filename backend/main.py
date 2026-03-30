"""
Backend FastAPI — Inteligencia de Prensa · Fundación Cibervoluntarios
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from parsers import parse_file
from rag import chat, delete_source, index_chunks, list_sources

# ── Config ─────────────────────────────────────────────────────────────────────

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 50
ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv"}


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ANTHROPIC_API_KEY:
        print("⚠️  ANTHROPIC_API_KEY no configurada. El chat no funcionará hasta configurarla.")
    yield


app = FastAPI(
    title="Inteligencia de Prensa — Cibervoluntarios",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modelos Pydantic ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    answer: str


# ── Endpoints API ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "api_key_configured": bool(ANTHROPIC_API_KEY)}


@app.post("/api/upload")
async def upload_file(file: Annotated[UploadFile, File()]):
    """Sube y indexa un fichero (PDF, Excel o CSV)."""
    if not file.filename:
        raise HTTPException(400, "Nombre de fichero vacío")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Formato no soportado: '{ext}'. Formatos válidos: PDF, XLSX, XLS, CSV",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"El fichero supera el límite de {MAX_UPLOAD_MB} MB")

    # Parsear
    try:
        chunks = parse_file(content, file.filename)
    except Exception as e:
        raise HTTPException(422, f"Error al procesar el fichero: {e}")

    if not chunks:
        raise HTTPException(422, "No se pudo extraer texto del fichero")

    # Indexar
    indexed = index_chunks(chunks)

    # Guardar copia local
    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(content)

    return {
        "filename": file.filename,
        "chunks_indexed": indexed,
        "size_mb": round(size_mb, 2),
        "message": f"'{file.filename}' indexado correctamente ({indexed} fragmentos)",
    }


@app.get("/api/documents")
def get_documents():
    """Lista los documentos indexados y su número de fragmentos."""
    return {"documents": list_sources()}


@app.delete("/api/documents/{filename}")
def remove_document(filename: str):
    """Elimina un documento del índice."""
    deleted = delete_source(filename)
    if deleted == 0:
        raise HTTPException(404, f"Documento '{filename}' no encontrado en el índice")
    # Eliminar fichero físico si existe
    phys = UPLOAD_DIR / filename
    if phys.exists():
        phys.unlink()
    return {"message": f"'{filename}' eliminado ({deleted} fragmentos borrados)"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Consulta conversacional con RAG sobre los documentos indexados."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            503,
            "La API key de Anthropic no está configurada. "
            "Añade ANTHROPIC_API_KEY en el fichero .env y reinicia el servidor.",
        )

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        answer = chat(messages, api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        raise HTTPException(500, f"Error al generar respuesta: {e}")

    return ChatResponse(answer=answer)


# ── Servir frontend ────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_ui():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
