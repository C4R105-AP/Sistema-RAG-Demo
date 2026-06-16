# Sistema RAG Universal

API de recuperación aumentada (RAG) multi-documento: PDF, DOCX, TXT y MD. Agnóstico de dominio — sin reglas por tipo de documento.

## Requisitos

- Python 3.11+
- [Ollama](https://ollama.com) (recomendado, local y gratuito) u OpenAI/Anthropic

## Instalación

```powershell
cd y:\_Doc_Tec_SOFT\_RAG_Ejemplo
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Configura `.env` (mínimo para Ollama):

```
LLM_TYPE=ollama
LLM_MODEL=llama3.2
```

Coloca documentos en `uploaded_docs/` y reindexa una vez:

```
POST http://localhost:8000/reindexar
```

## Arranque

```powershell
.\venv\Scripts\python.exe launcher.py
```

También válido: `.\venv\Scripts\python.exe api_rag.py` (redirige a `launcher.py`).

- Interfaz web: http://localhost:8000/app
- API docs: http://localhost:8000/docs

## Pipeline

```
Ingesta → chunking → embeddings → FAISS
  → similarity (k×4) → cross-encoder rerank → contexto → LLM
```

## Endpoints principales

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/preguntar` | Pregunta con respuesta del LLM |
| POST | `/buscar` | Chunks recuperados (sin LLM) |
| POST | `/resumir` | Resumen del corpus |
| POST | `/subir-documento` | Subir archivo |
| POST | `/reindexar` | Reconstruir índice |
| GET | `/estado` | Estado del sistema |

## Debug

En `.env`:

```
DEBUG_RAG=true
```

Muestra en consola similarity scores, rerank y orden final del contexto.

## Licencia

MIT — ver [LICENSE](LICENSE).
