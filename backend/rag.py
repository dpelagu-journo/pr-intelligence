"""
Motor RAG: indexación con ChromaDB + embeddings ONNX (fastembed)
y generación de respuestas con Claude (Anthropic API).
"""

import os
import uuid
from typing import Optional

import anthropic
import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

# ── Config ─────────────────────────────────────────────────────────────────────

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
COLLECTION_NAME = "press_intelligence"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 6  # chunks a recuperar por consulta
CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """Eres el asistente de inteligencia de prensa de la Fundación Cibervoluntarios, \
una ONG española dedicada a la inclusión digital.

Tu función es ayudar al equipo de comunicación y prensa a:
- Consultar el histórico de notas de prensa, argumentarios y dossieres
- Analizar registros de contactos con periodistas y medios
- Identificar patrones en los impactos de prensa conseguidos
- Redactar nuevas notas o pitches en el tono de la organización
- Sugerir estrategias de comunicación basadas en datos históricos

Responde SIEMPRE en español. Sé conciso y práctico. \
Cuando uses información de los documentos, cita explícitamente la fuente \
entre corchetes, por ejemplo: [Fuente: nota_prensa_2024.pdf, pág. 3] \
o [Fuente: registro_acciones.xlsx, filas 15-24].

Si no encuentras información suficiente en los documentos para responder, \
dilo claramente y sugiere qué datos adicionales podrían ayudar."""


# ── Singleton del modelo de embeddings ────────────────────────────────────────

_embed_model: Optional[TextEmbedding] = None


def _get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(EMBED_MODEL_NAME)
    return _embed_model


# ── ChromaDB ───────────────────────────────────────────────────────────────────

def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=VECTOR_STORE_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── API pública ────────────────────────────────────────────────────────────────

def index_chunks(chunks: list[dict]) -> int:
    """
    Indexa una lista de chunks en ChromaDB.
    Cada chunk: {"text": str, "metadata": dict}
    Devuelve el número de chunks indexados.
    """
    if not chunks:
        return 0

    model = _get_embed_model()
    collection = _get_collection()

    texts = [c["text"] for c in chunks]
    embeddings = list(model.embed(texts))

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [c.get("metadata", {}) for c in chunks]
    # ChromaDB requiere que los valores de metadata sean str/int/float/bool
    clean_metadatas = []
    for m in metadatas:
        clean_metadatas.append({k: str(v) for k, v in m.items()})

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=clean_metadatas,
    )
    return len(chunks)


def list_sources() -> list[dict]:
    """Devuelve los documentos indexados (nombre y nº de chunks)."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    sources: dict[str, int] = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "desconocido")
        sources[src] = sources.get(src, 0) + 1
    return [{"filename": k, "chunks": v} for k, v in sorted(sources.items())]


def delete_source(filename: str) -> int:
    """Elimina todos los chunks de un documento por su nombre."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    ids_to_delete = [
        result["ids"][i]
        for i, meta in enumerate(result["metadatas"])
        if meta.get("source") == filename
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Recupera los chunks más relevantes para una consulta."""
    model = _get_embed_model()
    collection = _get_collection()

    query_embedding = list(model.embed([query]))[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": round(1 - dist, 3),  # cosine similarity
        })
    return chunks


def _format_context(chunks: list[dict]) -> str:
    """Formatea los chunks recuperados como contexto para Claude."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        source = meta.get("source", "desconocido")
        detail = ""
        if "page" in meta:
            detail += f", pág. {meta['page']}"
        if "sheet" in meta:
            detail += f", hoja '{meta['sheet']}'"
        if "rows" in meta:
            detail += f", filas {meta['rows']}"
        parts.append(f"[Fragmento {i} — Fuente: {source}{detail}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def chat(
    messages: list[dict],
    api_key: str,
) -> str:
    """
    Realiza una consulta RAG completa:
    1. Recupera contexto relevante para el último mensaje del usuario
    2. Llama a Claude con el contexto + historial de conversación
    3. Devuelve la respuesta como string

    messages: lista de {"role": "user"|"assistant", "content": str}
    """
    if not messages:
        return "Por favor, escribe una pregunta."

    # Última pregunta del usuario para la búsqueda semántica
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    # Recuperar contexto
    chunks = retrieve(last_user_msg)

    # Construir mensajes para Claude
    claude_messages = []

    # Inyectar contexto en el primer turno o como mensaje de sistema adicional
    if chunks:
        context_text = _format_context(chunks)
        context_injection = (
            f"A continuación tienes fragmentos de documentos relevantes para responder "
            f"la pregunta. Úsalos como base y cita las fuentes.\n\n{context_text}"
        )
        # Prepend context to the first user message in history
        for m in messages:
            if m["role"] == "user":
                claude_messages.append({
                    "role": "user",
                    "content": f"{context_injection}\n\n---\n\nPregunta del usuario: {m['content']}",
                })
                break
        # Resto del historial
        first = True
        for m in messages:
            if first and m["role"] == "user":
                first = False
                continue
            claude_messages.append({"role": m["role"], "content": m["content"]})
    else:
        # Sin contexto RAG
        claude_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        claude_messages.insert(0, {
            "role": "user",
            "content": (
                "AVISO: No hay documentos indexados aún. "
                "Por favor, sube primero algunos documentos para poder responder con datos reales.\n\n"
                f"Pregunta: {last_user_msg}"
            ),
        })
        # Reemplazar solo el primer mensaje
        claude_messages = claude_messages[:1]

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=claude_messages,
    )
    return response.content[0].text
