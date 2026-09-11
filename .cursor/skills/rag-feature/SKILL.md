---
name: rag-feature
description: >-
  Implements RAG pipeline changes (retrieval, QA postprocess, translation, ingest)
  without domain hardcoding. Use when adding intents, fixing chat answers, changing
  chunking/rerank/translation, or extending the system for arbitrary documents.
---

# RAG feature changes

## Non-negotiables
1. Retrieval runs **before** any answer path (except translating an already-returned text).
2. No per-document phrase tables (IPC, Bizneo, etc.).
3. Prefer form-based citation scoring: query term, definitional shape, short norms, penalize TOC.
4. Keep layers: HTTP in `api.py`; logic in `retrieval` / `qa` / `llm` / `ingest`.

## Checklist
```text
Feature progress:
- [ ] 1. Locate layer (retrieval / qa / llm / ingest / web)
- [ ] 2. Implement without corpus-specific strings
- [ ] 3. Smoke 2–3 questions on /preguntar or /app
- [ ] 4. Run eval gate skill (rag-eval-gate) if pipeline touched
- [ ] 5. Update README / docs/PROMPT_ASISTENTE if behavior or .env changed
- [ ] 6. Commit only source + docs (no .env, PDFs, FAISS, rag.txt)
```

## Where to change what

| Need | Touch |
|------|--------|
| Ranking, TOC, context build | `rag/retrieval.py` |
| Intents, literal cite, cleanup, OOS | `rag/qa.py` |
| Ollama / Marian / Google translate | `rag/llm.py` + `.env.example` |
| Chunk size, prompts, flags | `rag/config.py` |
| Endpoints / source snippets | `rag/api.py` |
| UI translate button | `web/interfaz_web.html` |
| Harness cases | `tests/eval_rag.py` |

## Common pitfalls
- **Same PDF sentence, two concepts** (Aspect + Area): sanitize context / strip foreign thresholds; do not special-case filenames.
- **Overlap** duplicates edges; rarely the root cause of concept mix.
- **ES question / EN PDF**: keep multilingual embeddings; tell users to include English technical terms when needed.
- **LLM loops**: dedupe paragraphs in postprocess; keep `num_predict` modest and `repeat_penalty` ≥ ~1.4.
- Skipping FAISS for “known” questions breaks agnosticism — do not reintroduce.

## Smoke (optional, fast)
```powershell
# OOS
# Cita: Copia literalmente la definición de Area Ratio.
# Bizneo: ¿Cómo se solicitan las vacaciones?
```

## Afterward
Use skill `rag-eval-gate` before calling the change done if retrieval/QA/prompts changed.
