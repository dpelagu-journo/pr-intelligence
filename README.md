# Inteligencia de Prensa · Fundación Cibervoluntarios

Herramienta interna de inteligencia de prensa con chat conversacional en español.
El equipo de comunicación puede subir documentos (notas de prensa, Excels, CSVs) y hacer preguntas en lenguaje natural.

## Arquitectura

```
Frontend (HTML/JS)  →  Backend (FastAPI)  →  RAG (ChromaDB + sentence-transformers)  →  Claude API
```

- **Parser**: extrae texto de PDF, Excel y CSV en chunks con metadata de origen
- **Embeddings**: modelo multilingüe local `paraphrase-multilingual-MiniLM-L12-v2` (sin coste adicional)
- **Vector store**: ChromaDB persistente en disco (`data/vector_store/`)
- **LLM**: Claude claude-sonnet-4-6 vía API de Anthropic
- **Citas**: cada respuesta incluye `[Fuente: nombre_fichero, pág. X]`

## Instalación

### 1. Requisitos previos
- Python 3.11+
- Clave de API de Anthropic ([console.anthropic.com](https://console.anthropic.com/))

### 2. Configurar variables de entorno

```bash
cd pr-intelligence
cp .env.example .env
# Edita .env y añade tu ANTHROPIC_API_KEY
```

### 3. Instalar dependencias

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> La primera vez que arrances, `sentence-transformers` descarga el modelo (~120 MB). Solo ocurre una vez.

### 4. Arrancar el servidor

```bash
# Desde el directorio backend/
uvicorn main:app --reload --port 8000
```

Abre el navegador en **http://localhost:8000**

## Uso

1. **Sube documentos** desde el panel lateral (arrastrar o clic)
   - Notas de prensa: `.pdf`
   - Registro de acciones, contactos de periodistas: `.xlsx` / `.csv`
   - Impactos de Simbiu: `.csv`

2. **Haz preguntas** en el chat:
   - *"¿Qué hemos dicho sobre brecha digital en mayores?"*
   - *"¿Qué periodistas han recogido más nuestras notas?"*
   - *"Redáctame una nota sobre [tema] en nuestro tono habitual"*

3. Las respuestas citan siempre la fuente entre corchetes: `[Fuente: nota_2024.pdf, pág. 3]`

## API REST

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/api/health` | Estado del servidor |
| `POST` | `/api/upload` | Sube e indexa un fichero |
| `GET` | `/api/documents` | Lista documentos indexados |
| `DELETE` | `/api/documents/{filename}` | Elimina un documento |
| `POST` | `/api/chat` | Consulta conversacional |

Documentación interactiva en: **http://localhost:8000/docs**

## Roadmap (próximas fases)

- [ ] Integración con Google Drive (descarga automática)
- [ ] Sincronización automática de Excel de contactos
- [ ] Soporte multi-hoja con detección de tipo (contactos / acciones / impactos)
- [ ] Filtros por fecha y geografía en el chat
- [ ] Autenticación básica multiusuario
- [ ] Panel de análisis de impactos
