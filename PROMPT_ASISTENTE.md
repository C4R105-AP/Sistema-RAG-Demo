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

## Estado de evaluación: 10/10 PASS (2026-09-01)

Fuente de verdad: `eval_baseline.json`. Hay que regenerarlo con `python eval_rag.py --baseline` tras esta corrida.

### Tests
Los 10 casos del harness pasan: Area Ratio, cita literal, páginas solder paste, Aspect Ratio, BGA, vacaciones, QR, credenciales, aviso QR y descansos.

### Cómo se cerraron los FAIL anteriores
- **Aspect Ratio**: retrieval prioriza el bigrama `aspect ratio`; el LLM ya no mezcla el umbral `0.66`.
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
python eval_rag.py              # 10 tests con Ollama
python eval_rag.py --baseline   # guardar eval_baseline.json
python eval_rag.py --compare    # fallar si un PASS pasa a FAIL
```

## Archivos de interés

- `api_rag.py` — pipeline RAG y API
- `launcher.py` — arranque (bind `127.0.0.1` por defecto)
- `interfaz_web.html` — UI en `/app`
- `eval_rag.py` — harness de 10 tests
- `eval_baseline.json` — último baseline versionado

## Seguridad mínima (aplicada)

- `RAG_HOST=127.0.0.1` por defecto
- Upload con `basename` y comprobación de ruta
- Chat con `textContent` (sin `innerHTML` de pregunta/respuesta/chunks)
- `DELETE /limpiar` deshabilitado salvo `X-RAG-Admin-Token` = `RAG_ADMIN_TOKEN`

**Generado**: 10/09/2026
**Objetivo**: mantener 10/10 con `eval_rag.py --compare`

Además del harness: rechazo rápido si la query no está en el índice; citas literales desde el corpus; nota de páginas sin listado del LLM; botón *Traducir* solo en respuestas en inglés.
