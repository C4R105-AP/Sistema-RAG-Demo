# Prompt para Asistente Externo - Sistema RAG

## Contexto del Proyecto

Sistema RAG (Retrieval-Augmented Generation) multi-dominio:
- **Backend**: FastAPI + LangChain
- **Vectorstore**: FAISS con embeddings locales (sentence-transformers)
- **LLM**: Ollama con llama3.2 (CPU local)
- **Traducción UI**: Google Translate por defecto (`deep-translator`); respaldo MarianMT / DeepL / Ollama
- **Corpus demo**: 3 PDF (~195 chunks)
  - IPC-7525A - Stencil Design Guidelines (inglés)
  - IPC-7711 - Rework of Electronic Assemblies (inglés)
  - Manual_Bizneo.pdf (español)

## Estado de evaluación

Baseline versionado: `tests/eval_baseline.json` (12 casos, 2026-09-10).

Tras cambios de pipeline, ejecutar:

```bash
python tests/eval_rag.py --compare
# si 12/12 y sin regresiones:
python tests/eval_rag.py --baseline
```

### Tests del harness
Area Ratio, cita literal Area Ratio, páginas solder paste (P/R), Aspect Ratio, BGA, vacaciones, QR, credenciales, aviso QR, descansos, OOS paella, OOS caballo.

### Principios de diseño (no romper)
1. **Siempre retrieval primero** — no hay atajos que salten FAISS/BM25 por documento concreto (salvo traducción de un texto ya respondido).
2. **Agnóstico de dominio** — puntuación de citas por forma (definición/norma + término de la query), no por frases IPC/Bizneo hardcodeadas.
3. **Fuera de ámbito** — tras consultar el índice: si faltan términos sustanciales → mensaje fijo.
4. **Fuentes** — sin tablas de contenidos; snippet centrado en el término; máximo ~4.
5. **Postproceso** — quitar `RESPUESTA:`, deduplicar bucles del LLM, anexar PDFs en resúmenes, glosa aperture→apertura si el contexto lo trae y la pregunta va en español.

### Problemas conocidos / mitigados
- **Aspect vs Area Ratio** en la misma oración del PDF: el contexto se sanea y se evitan umbrales cruzados; el LLM a veces sigue mezclando tablas.
- **Aviso QR literal**: la cita debe preferir la norma corta (*Debe ficharse…*); si falla el scoring, puede salir otra frase del mismo chunk.
- **Descansos**: el manual mezcla fichaje QR en la misma sección; se filtra por término, pero el LLM puede divagar.
- **ES query / EN docs**: embeddings multilingües ayudan; BM25 no. Conviene el término técnico en inglés en la pregunta.

## Arquitectura

### Pipeline de retrieval
1. BM25 + semántica → RRF (k=60)
2. Ranking léxico del término más discriminante
3. Cuota por documento → recorte `top_n`
4. Cross-encoder rerank
5. Dedup SequenceMatcher 0.85
6. Top-k con cobertura del término; TOC al final
7. Contexto → LLM (o cita / nota de páginas)

### Configuración LLM
```python
ChatOllama(
    model="llama3.2",
    temperature=0,
    num_predict=350,
    repeat_penalty=1.45,
    keep_alive="30m",
)
```

### Traducción (`.env`)
```
TRANSLATE_BACKEND=google   # deepl | marian | ollama
# DEEPL_API_KEY=...
# TRANSLATE_MODEL=Helsinki-NLP/opus-mt-en-es
```

## Cómo ejecutar

```powershell
ollama serve
.\venv\Scripts\python.exe launcher.py
# http://localhost:8000/app

.\venv\Scripts\python.exe tests\eval_rag.py --compare
```

## Archivos de interés

- `rag/api.py` — FastAPI
- `rag/retrieval.py` / `rag/qa.py` — pipeline y postproceso
- `rag/llm.py` — embeddings, ChatOllama, `traducir_en_es`
- `rag/ingest.py` / `rag/store.py` — ingesta e índice
- `launcher.py` — arranque (`127.0.0.1`)
- `web/interfaz_web.html` — UI
- `tests/eval_rag.py` + `tests/eval_baseline.json`
- `.env.example` — plantilla de configuración
- `.cursor/rules/rag-standards.mdc` — estándares del agente
- `.cursor/skills/rag-eval-gate` — harness `--compare` / `--baseline`
- `.cursor/skills/rag-feature` — cambios de pipeline sin hardcodear dominio

## Seguridad mínima

- `RAG_HOST=127.0.0.1` por defecto
- Upload con `basename` y ruta acotada
- Chat con `textContent`
- `DELETE /limpiar` solo con `X-RAG-Admin-Token`

**Actualizado**: 11/09/2026  
**Objetivo**: mantener 12/12 con `eval_rag.py --compare` y respuestas útiles en `/app` sin reglas por documento.
