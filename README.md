![Platform](https://img.shields.io/badge/platform-Windows-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# Sistema RAG Universal

API y chat para preguntar sobre documentos (PDF, DOCX, TXT, MD). El sistema recupera fragmentos relevantes del índice y genera la respuesta con un LLM. Es agnóstico de dominio: no hay reglas fijas por tipo de documento.

Por defecto corre **en local** (Ollama + embeddings en CPU). No hace falta exponer la API a la red.

## Qué hace

1. **Ingesta** — extrae texto de los archivos en `uploaded_docs/`, limpia artefactos de PDF y parte en chunks (~900 caracteres, solape 200).
2. **Índice** — embeddings multilingües (sentence-transformers) + FAISS en disco (`vectorstore_faiss/`).
3. **Retrieval híbrido** — BM25 + búsqueda semántica, fusión RRF, cuota por documento, rerank con cross-encoder y deduplicación.
4. **Generación** — el LLM solo ve el contexto recuperado. Citas literales y listados de páginas se resuelven sobre el corpus, no inventando.
5. **Chat** — interfaz en `/app`: fuentes plegadas y botón *Traducir al castellano* solo si la respuesta está en inglés.

```
Pregunta
   │
   ├─ ¿Palabras sustanciales ausentes del índice?  →  rechazo inmediato (sin FAISS ni LLM)
   │
   ▼
BM25 + semántica (+ léxico del término más selectivo)
   │
   ▼
RRF → cuota por documento → rerank → top-k
   │
   ├─ Copia / «qué dice exactamente»  →  oración extraída del corpus
   ├─ «¿En qué páginas…?»             →  nota exhaustiva
   └─ Resto                           →  contexto + LLM
```

## Requisitos (modo fuente)

- Windows 10/11 o equivalente, Python 3.11+
- [Ollama](https://ollama.com) con un modelo local (p. ej. `ollama pull llama3.2`)
- Opcional: clave de OpenAI o Anthropic en `.env` (nunca en el repositorio)

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
```

Si Ollama no está instalado o no responde, el arranque cae a **modo demo** (`fake`): se puede buscar en el índice, pero no hay generación real.

Coloca los PDF/DOCX/TXT/MD en `uploaded_docs/` y reconstruye el índice una vez:

```
POST http://localhost:8000/reindexar
```

o súbelos desde la pestaña de la interfaz.

## Arranque

```powershell
.\venv\Scripts\python.exe launcher.py
```

También válido: `.\venv\Scripts\python.exe api_rag.py` (redirige al launcher).

- Interfaz: http://localhost:8000/app
- API: http://localhost:8000/docs
- Bind por defecto: `127.0.0.1:8000` (solo esta máquina). Para escuchar en todas las interfaces, `RAG_HOST=0.0.0.0` en `.env`.

## Interfaz (`/app`)

| Pestaña | Uso |
|---------|-----|
| Chat | Preguntas sobre el corpus. Las fuentes salen plegadas (`Fuentes (N)`). |
| Subir | Arrastrar archivos e indexarlos. |
| Resumir | Resumen del material indexado. |

Tras una respuesta **en inglés**, aparece *Traducir al castellano* (no en respuestas que ya están en español). El chat no tiene pestaña de traducción: o usas ese botón, o en el mismo hilo escribes «tradúcela» (seguimiento de la última respuesta; se pierde al recargar).

Preguntas sin ninguna palabra del índice (p. ej. una receta o un tema ajeno) se rechazan al instante, sin embeddings ni LLM.

## Pipeline de retrieval (detalle)

1. BM25 y similitud FAISS (en corpus pequeños se usan todos los chunks).
2. Ranking léxico del término o bigrama más discriminante de la pregunta.
3. Fusión RRF (`RRF_K=60`).
4. Cuota por documento **antes** de recortar a `top_n`.
5. Cross-encoder (`ms-marco-MiniLM-L-6-v2`) y dedup (SequenceMatcher 0.85).
6. Top-k (`DEFAULT_K=8`) priorizando cobertura de ese término.

Intents especiales:

- **Literal / «qué dice exactamente»** — se copia la oración del corpus (glosario, avisos, normas).
- **Páginas / «dónde aparece»** — recuento léxico de páginas; no se deja que el LLM invente un subconjunto.
- **Resumen de apartados** — el contexto se filtra al término pedido; se anexan los PDF usados.

## Evaluación

Única puerta: `eval_rag.py` (10 casos; requiere Ollama).

```powershell
.\venv\Scripts\python.exe eval_rag.py
.\venv\Scripts\python.exe eval_rag.py --baseline
.\venv\Scripts\python.exe eval_rag.py --compare
```

El último baseline versionado está en `eval_baseline.json`. Un `--compare` con código 0 significa que no hay regresiones respecto a ese archivo.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/preguntar` | Pregunta + respuesta del LLM (acepta seguimiento: pregunta/respuesta anterior) |
| POST | `/buscar` | Chunks recuperados, sin LLM |
| POST | `/resumir` | Resumen del corpus |
| POST | `/subir-documento` | Subir e indexar un archivo |
| POST | `/reindexar` | Reconstruir el índice |
| GET | `/estado` | Estado y número de chunks |
| GET | `/app` | Interfaz |

`DELETE /limpiar` no es público: hace falta el header `X-RAG-Admin-Token` igual a `RAG_ADMIN_TOKEN` en `.env`. Si el token no está definido, responde 403.

## Configuración

Copia `.env.example` → `.env`. El fichero `.env` está en `.gitignore` (no lo subas).

| Variable | Rol |
|----------|-----|
| `LLM_TYPE` | `ollama` (recomendado), `openai`, `anthropic`, `fake` |
| `LLM_MODEL` | Modelo Ollama, p. ej. `llama3.2` |
| `RAG_HOST` / `RAG_PORT` | Bind del servidor |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Tamaño de fragmento |
| `DEFAULT_K` | Chunks finales al LLM |
| `DEBUG_RAG` | Traza de scores en consola |
| `RAG_ADMIN_TOKEN` | Solo si quieres habilitar borrar índice |

No pegues claves de API ni tokens en el README, en issues ni en commits. Si usas un proveedor de pago, deja la clave solo en `.env`.

## Seguridad (mínimo local)

- Escucha en localhost por defecto.
- Nombre de archivo al subir: solo el basename, ruta acotada a `uploaded_docs/`.
- El chat pinta pregunta, respuesta y chunks con `textContent` (no HTML inyectado).
- CORS limitado a localhost.
- Sin Ollama, no se llama a un endpoint remoto: modo demo.

El índice FAISS se carga con deserialización local (pickle). Trátalo como dato de confianza de tu máquina; no abras índices de terceros.

## Archivos principales

| Archivo | Función |
|---------|---------|
| `launcher.py` | Arranque (venv, puerto, navegador) |
| `api_rag.py` | Ingesta, retrieval, LLM y API |
| `interfaz_web.html` | UI |
| `eval_rag.py` | Harness de regresión |
| `.env.example` | Plantilla de configuración |

`uploaded_docs/` y `vectorstore_faiss/` no van al git: se regeneran en cada entorno.

## Licencia

MIT — ver [LICENSE](LICENSE).
