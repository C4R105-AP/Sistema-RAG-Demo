# Prompt para Asistente Externo - Sistema RAG

## Contexto del Proyecto

Sistema RAG (Retrieval-Augmented Generation) multi-dominio:
- **Backend**: FastAPI + LangChain
- **Vectorstore**: FAISS con embeddings locales (sentence-transformers)
- **LLM**: Ollama con llama3.2 (CPU local)
- **Corpus**: 3 documentos PDF (~195 chunks)
  - IPC-7525A - Stencil Design Guidelines (inglés)
  - IPC-7711 - Rework of Electronic Assemblies (inglés)
  - Manual_Bizneo.pdf (español)

## Estado de evaluación: 12/12 PASS (2026-09-10)

Fuente de verdad: `eval_baseline.json`. Regenerarlo con `python tests/eval_rag.py --baseline` tras un 12/12.

### Tests
Los 12 casos del harness: Area Ratio, cita literal, páginas solder paste (P/R de la nota 📄), Aspect Ratio, BGA, vacaciones, QR, credenciales, aviso QR, descansos, y 2 OOS (paella, caballo).

### Cómo se cerraron los FAIL anteriores
- **Aspect Ratio**: retrieval prioriza el bigrama `aspect ratio`; postproceso veta `>0.66 for area ratio` si la pregunta no pide Area Ratio.
- **Solder paste (páginas)**: la precisión se mide sobre las páginas de la nota (AND de términos de tema), no sobre el léxico OR de la pregunta.
- **OOS**: paella/caballo deben devolver el rechazo de ámbito, sin buscar en FAISS.
- **BGA**: cuota por documento + cobertura de `bga`; si el resumen no nombra los PDFs, se anexan (mismo patrón que la nota de páginas exhaustivas). Llama 3.2 ignoraba la instrucción de citar fuentes.
- **Area Ratio (regresión)**: si la pregunta está en español y el contexto trae *aperture/walls*, se añade la glosa *apertura/paredes*. El modelo no traducía de forma estable.

## Arquitectura actual

### Pipeline de retrieval
1. BM25 + semántica → RRF (k=60)
2. Ranking léxico del término más discriminante de la query (si no es intent exhaustivo/literal)
3. Cuota por documento sobre el ranking completo, después recorte a `top_n`
4. Cross-encoder rerank
5. Dedup SequenceMatcher 0.85
6. Top-k con cobertura del término discriminante
7. Contexto → LLM

### Configuración LLM
```python
ChatOllama(
    model="llama3.2",
    temperature=0,
    num_predict=500,
    repeat_penalty=1.3,
)
```

## Cómo ejecutar los tests

Única puerta de evaluación: `eval_rag.py`.

```bash
python tests/eval_rag.py              # 12 tests con Ollama
python tests/eval_rag.py --baseline   # guardar eval_baseline.json
python tests/eval_rag.py --compare    # fallar si un PASS pasa a FAIL
```

## Archivos de interés

- `rag/api.py` — FastAPI (endpoints)
- `rag/retrieval.py` / `rag/qa.py` — pipeline y postproceso
- `rag/ingest.py` / `rag/store.py` / `rag/llm.py` — ingesta, índice y modelos
- `launcher.py` — arranque (bind `127.0.0.1` por defecto)
- `web/interfaz_web.html` — UI en `/app`
- `tests/eval_rag.py` — harness de 12 tests
- `tests/eval_baseline.json` — último baseline versionado

## Seguridad mínima (aplicada)

- `RAG_HOST=127.0.0.1` por defecto
- Upload con `basename` y comprobación de ruta
- Chat con `textContent` (sin `innerHTML` de pregunta/respuesta/chunks)
- `DELETE /limpiar` deshabilitado salvo `X-RAG-Admin-Token` = `RAG_ADMIN_TOKEN`

**Generado**: 10/09/2026
**Objetivo**: mantener 12/12 con `eval_rag.py --compare`

Además del harness: rechazo rápido si la query no está en el índice; citas literales desde el corpus; nota de páginas sin listado del LLM; botón *Traducir* solo en respuestas en inglés.
