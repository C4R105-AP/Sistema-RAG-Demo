![Platform](https://img.shields.io/badge/platform-Windows-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# Sistema RAG Universal

API y chat para preguntar sobre documentos (PDF, DOCX, TXT, MD). El sistema recupera fragmentos relevantes del índice y genera la respuesta con un LLM. Es **agnóstico de dominio**: no hay reglas fijas por tipo de documento ni glosas hardcodeadas del corpus demo.

Por defecto corre **en local** (Ollama + embeddings en CPU). La API escucha solo en `127.0.0.1`.

## Qué hace

1. **Ingesta** — extrae texto de `data/uploaded_docs/`, limpia artefactos de PDF y parte en chunks (~900 caracteres, solape 200).
2. **Índice** — embeddings multilingües (sentence-transformers) + FAISS en disco (`data/vectorstore_faiss/`).
3. **Retrieval híbrido** — BM25 + semántica, fusión RRF, cuota por documento, rerank con cross-encoder y deduplicación. **Toda pregunta pasa primero por el índice** (también las fuera de ámbito).
4. **Generación** — el LLM solo ve el contexto recuperado. Citas literales y listados de páginas se resuelven sobre chunks, no inventando.
5. **Chat** — interfaz en `/app`: fuentes plegadas (sin índices TOC) y botón *Traducir al castellano* (Google Translate por defecto).

```
Pregunta
   │
   ▼
BM25 + semántica (+ léxico del término más selectivo)
   │
   ▼
RRF → cuota por documento → rerank → top-k
   │
   ├─ ¿Términos sustanciales ausentes del índice? → rechazo de ámbito
   ├─ «¿En qué páginas…?»             → nota exhaustiva 📄
   ├─ Copia / «qué dice exactamente»  → oración elegida de los chunks
   └─ Resto                           → contexto + LLM (+ limpieza)
```

## Requisitos (modo fuente)

- Windows 10/11 o equivalente, Python 3.11+
- [Ollama](https://ollama.com) con un modelo local (p. ej. `ollama pull llama3.2`)
- Internet solo si usas `TRANSLATE_BACKEND=google` (por defecto) o DeepL
- Opcional: clave de OpenAI, Anthropic o DeepL en `.env` (nunca en el repositorio)

Hay un ejecutable portable en [Releases](../../releases/latest) para quien no quiera instalar Python. El desarrollo y la evaluación se hacen desde este código fuente.

## Instalación

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

En `.env` basta con (valores de ejemplo, sin secretos):

```
LLM_TYPE=ollama
LLM_MODEL=llama3.2
RAG_HOST=127.0.0.1
TRANSLATE_BACKEND=google
```

Si Ollama no está instalado o no responde, el arranque cae a **modo demo** (`fake`): se puede buscar en el índice, pero no hay generación real.

Coloca los PDF/DOCX/TXT/MD en `data/uploaded_docs/` y reconstruye el índice una vez:

```
POST http://localhost:8000/reindexar
```

o súbelos desde la pestaña de la interfaz.

## Arranque

```powershell
# 1) Ollama (otra ventana o servicio)
ollama serve
ollama pull llama3.2

# 2) API + chat
.\venv\Scripts\python.exe launcher.py
```

También válido: `.\venv\Scripts\python.exe api_rag.py` (compatibilidad; redirige al launcher).

- Interfaz: http://localhost:8000/app
- API: http://localhost:8000/docs
- Bind por defecto: `127.0.0.1:8000`. Para todas las interfaces: `RAG_HOST=0.0.0.0` en `.env`.

## Interfaz (`/app`)

| Pestaña | Uso |
|---------|-----|
| Chat | Preguntas sobre el corpus. Las fuentes salen plegadas (`Fuentes (N)`), recortadas al término. |
| Subir | Arrastrar archivos e indexarlos. |
| Resumir | Resumen del material indexado. |

Tras una respuesta **en inglés**, aparece *Traducir al castellano*. El botón usa el backend de traducción configurado (Google por defecto; ~1 s). También puedes escribir «tradúcela» como seguimiento.

Preguntas sin ninguna palabra del índice (p. ej. una receta ajena) se rechazan **después** de consultar el índice, con mensaje fijo de fuera de ámbito.

## Pipeline de retrieval (detalle)

1. BM25 y similitud FAISS (en corpus pequeños se usan todos los chunks).
2. Ranking léxico del término o bigrama más discriminante de la pregunta.
3. Fusión RRF (`RRF_K=60`).
4. Cuota por documento **antes** de recortar a `top_n`.
5. Cross-encoder (`ms-marco-MiniLM-L-6-v2`) y dedup (SequenceMatcher 0.85).
6. Top-k (`DEFAULT_K=8`) priorizando cobertura de ese término; se retrasan chunks tipo **tabla de contenidos**.
7. Contexto enfocado al término; en Aspect Ratio se limpia el umbral ajeno `0.66` del contexto cuando la query no pide Area Ratio.

Intents:

- **Literal / «qué dice exactamente»** — oración puntuada de los chunks (forma de definición/norma + término; sin bonus por documento concreto).
- **Páginas / «dónde aparece»** — nota 📄 por AND de términos de tema.
- **Resumen de apartados** — contexto filtrado al término; se anexan los PDF usados; se deduplican bucles del LLM.

## Traducción

| `TRANSLATE_BACKEND` | Uso |
|---------------------|-----|
| `google` (default) | Rápido, online (`deep-translator`) |
| `deepl` | Requiere `DEEPL_API_KEY` |
| `marian` | Offline: `Helsinki-NLP/opus-mt-en-es` |
| `ollama` | Mismo LLM local (más lento en CPU) |

Cascada de respaldo: si el backend falla → MyMemory / Marian / Ollama según el caso.

## Idioma de la pregunta vs documentos

Los embeddings son **multilingües**, pero BM25 es léxico. Con PDFs en inglés conviene incluir el **término técnico en inglés** en la pregunta (*Area Ratio*, *BGA*, *solder paste*), aunque el resto vaya en castellano. Los manuales en español (p. ej. Bizneo) van bien con preguntas en castellano.

## Evaluación

Única puerta: `eval_rag.py` (12 casos; requiere Ollama). Incluye 2 fuera de corpus y precisión/recall de páginas en solder paste.

```powershell
.\venv\Scripts\python.exe tests\eval_rag.py
.\venv\Scripts\python.exe tests\eval_rag.py --baseline
.\venv\Scripts\python.exe tests\eval_rag.py --compare
```

Baseline versionado: `tests/eval_baseline.json`. `--compare` con código 0 = sin regresiones PASS→FAIL.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/preguntar` | Pregunta + respuesta (seguimiento: pregunta/respuesta anterior) |
| POST | `/buscar` | Chunks recuperados, sin LLM |
| POST | `/resumir` | Resumen del corpus |
| POST | `/subir-documento` | Subir e indexar |
| POST | `/reindexar` | Reconstruir el índice |
| GET | `/estado` | Estado y número de chunks |
| GET | `/app` | Interfaz |

`DELETE /limpiar` exige `X-RAG-Admin-Token` = `RAG_ADMIN_TOKEN`. Sin token en `.env` → 403.

## Configuración

Copia `.env.example` → `.env`. El `.env` está en `.gitignore`.

| Variable | Rol |
|----------|-----|
| `LLM_TYPE` | `ollama` (recomendado), `openai`, `anthropic`, `fake` |
| `LLM_MODEL` | Modelo Ollama, p. ej. `llama3.2` |
| `RAG_HOST` / `RAG_PORT` | Bind del servidor |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Tamaño de fragmento |
| `DEFAULT_K` | Chunks finales al LLM |
| `TRANSLATE_BACKEND` | `google` / `deepl` / `marian` / `ollama` |
| `TRANSLATE_MODEL` | Modelo Marian u Ollama para traducir |
| `DEEPL_API_KEY` | Solo con `TRANSLATE_BACKEND=deepl` |
| `DEBUG_RAG` | Traza de scores en consola |
| `RAG_ADMIN_TOKEN` | Habilita borrar índice |

No pegues claves ni tokens en el README, issues ni commits.

## Seguridad (mínimo local)

- Escucha en localhost por defecto.
- Upload: basename y ruta acotada a `uploaded_docs/`.
- Chat con `textContent` (sin HTML inyectado).
- CORS limitado a localhost.
- Sin Ollama → modo demo, sin llamada remota de generación.

El índice FAISS usa deserialización local (pickle): solo índices de confianza en tu máquina.

## Estructura del repositorio

```
.
├── launcher.py          # Arranque
├── api_rag.py           # Compatibilidad: reexporta rag.api
├── rag/                 # Paquete de la aplicación
│   ├── api.py           # FastAPI
│   ├── config.py        # Constantes y prompts
│   ├── ingest.py        # PDF/DOCX y chunking
│   ├── retrieval.py     # BM25 + RRF + rerank
│   ├── qa.py            # Cadena QA y postproceso
│   ├── llm.py           # Embeddings, LLM y traducción
│   ├── store.py         # Carga/reindexado FAISS
│   └── paths.py         # Rutas (data/, web/)
├── web/                 # Interfaz estática
├── tests/               # eval_rag.py y baseline
├── docs/                # Notas para desarrollo
├── data/                # Runtime (no se versiona índice ni PDF)
│   ├── uploaded_docs/
│   └── vectorstore_faiss/
├── .env.example
└── requirements.txt
```

## Limitaciones conocidas

- Conceptos distintos en la **misma oración del PDF** (p. ej. Aspect y Area Ratio) pueden mezclarse en la respuesta; el postproceso mitiga umbrales cruzados, no reescribe el documento.
- El solape de chunks (200) puede duplicar bordes; no es la causa principal de esa mezcla.
- `llama3.2` en CPU es lento en resúmenes (~1–2 min); las citas y la traducción Google son rápidas.

## Licencia

MIT — ver [LICENSE](LICENSE).
