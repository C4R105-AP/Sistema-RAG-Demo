# Sistema RAG Universal

API de recuperación aumentada (RAG) multi-documento: PDF, DOCX, TXT y MD. Agnóstico de dominio — sin reglas por tipo de documento.

**Estado de evaluación (baseline 30/06/2026): 8/10 PASS.** Fallan Aspect Ratio (el LLM mezcla el umbral 0.66 de Area Ratio) y BGA (el retrieval no incluía IPC-7525A). El resto de casos (vacaciones, QR, credenciales, Area Ratio, páginas) pasan.

## Requisitos

- Python 3.11+
- [Ollama](https://ollama.com) (recomendado, local y gratuito) u OpenAI/Anthropic

## Instalación

```powershell
cd C:\SCRIPTS\_RAG_Ejemplo
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Configura `.env` (mínimo para Ollama):

```
LLM_TYPE=ollama
LLM_MODEL=llama3.2
RAG_HOST=127.0.0.1
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

El servidor escucha en `127.0.0.1:8000` (no en toda la LAN). Para bind externo, define `RAG_HOST=0.0.0.0` en `.env`.

- Interfaz web: http://localhost:8000/app
- API docs: http://localhost:8000/docs

## Pipeline

```
Ingesta → chunking → embeddings → FAISS
  → BM25 + semántica (+ léxico del término discriminante) → RRF
  → cuota por documento (antes del corte top_n)
  → cross-encoder rerank → cobertura del término discriminante
  → contexto → LLM
```

## Evaluación

La única puerta de evaluación es `eval_rag.py` (10 casos contra Ollama):

```powershell
.\venv\Scripts\python.exe eval_rag.py
.\venv\Scripts\python.exe eval_rag.py --baseline
.\venv\Scripts\python.exe eval_rag.py --compare
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

`DELETE /limpiar` exige el header `X-RAG-Admin-Token` igual a `RAG_ADMIN_TOKEN`. Si el token no está definido, el endpoint responde 403.

## Debug

En `.env`:

```
DEBUG_RAG=true
```

Muestra en consola similarity scores, rerank y orden final del contexto.

## Licencia

MIT — ver [LICENSE](LICENSE).
