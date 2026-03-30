"""
Parsers para documentos: PDF, Excel (XLSX/XLS) y CSV.
Devuelven listas de chunks con metadata de origen.
"""

import io
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber


# ── Utilidades ────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Elimina espacios múltiples y líneas en blanco consecutivas."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Divide texto largo en chunks solapados para mejor recuperación."""
    words = text.split()
    chunks = []
    i = 0
    chunk_index = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text.strip()) > 30:  # ignorar chunks casi vacíos
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index,
                },
            })
            chunk_index += 1
        i += chunk_size - overlap
    return chunks


# ── PDF ───────────────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Extrae texto de un PDF página a página.
    Cada chunk corresponde a una página (o fracción si es muy larga).
    """
    chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            text = _clean(raw)
            if not text:
                continue
            # Si la página es muy larga, sub-chunkeamos
            if len(text.split()) > 800:
                sub = _chunk_text(text, source=filename, chunk_size=600, overlap=80)
                for c in sub:
                    c["metadata"]["page"] = page_num
                    c["metadata"]["source"] = filename
                chunks.extend(sub)
            else:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                        "chunk_index": len(chunks),
                    },
                })
    return chunks


# ── Excel / CSV ───────────────────────────────────────────────────────────────

def _df_to_chunks(df: pd.DataFrame, filename: str) -> list[dict]:
    """
    Convierte un DataFrame en chunks de texto.
    Estrategia: cada fila → un string "Campo: valor | Campo: valor ..."
    Agrupa filas en bloques de 10 para no crear demasiados chunks pequeños.
    """
    df = df.dropna(how="all").fillna("")
    # Normalizar nombres de columnas
    df.columns = [str(c).strip() for c in df.columns]

    row_texts = []
    for _, row in df.iterrows():
        parts = [f"{col}: {str(val).strip()}" for col, val in row.items() if str(val).strip()]
        if parts:
            row_texts.append(" | ".join(parts))

    # Agrupar de 10 en 10 filas
    chunks = []
    batch_size = 10
    for i in range(0, len(row_texts), batch_size):
        batch = row_texts[i : i + batch_size]
        text = "\n".join(batch)
        chunks.append({
            "text": text,
            "metadata": {
                "source": filename,
                "rows": f"{i + 1}-{min(i + batch_size, len(row_texts))}",
                "chunk_index": len(chunks),
            },
        })
    return chunks


def parse_excel(file_bytes: bytes, filename: str) -> list[dict]:
    """Parsea todas las hojas de un Excel."""
    chunks = []
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    for sheet in xls.sheet_names:
        df = xls.parse(sheet, dtype=str)
        sheet_chunks = _df_to_chunks(df, filename)
        for c in sheet_chunks:
            c["metadata"]["sheet"] = sheet
        chunks.extend(sheet_chunks)
    return chunks


def parse_csv(file_bytes: bytes, filename: str) -> list[dict]:
    """Parsea un CSV."""
    # Intentar detectar separador automáticamente
    sample = file_bytes[:2048].decode("utf-8", errors="replace")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, dtype=str)
    return _df_to_chunks(df, filename)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def parse_file(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Punto de entrada único. Devuelve lista de chunks:
      [{"text": str, "metadata": {"source": str, ...}}, ...]
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_bytes, filename)
    elif ext in (".xlsx", ".xls"):
        return parse_excel(file_bytes, filename)
    elif ext == ".csv":
        return parse_csv(file_bytes, filename)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Soportados: PDF, XLSX, XLS, CSV")
