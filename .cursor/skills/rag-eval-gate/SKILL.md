---
name: rag-eval-gate
description: >-
  Runs the RAG regression harness (eval_rag.py --compare / --baseline) with Ollama,
  interprets PASS/FAIL and page P/R. Use when changing retrieval, QA, prompts, chunking,
  after fixing chat quality, or when the user asks to evaluate, compare baseline, or
  regenerate eval_baseline.json.
---

# RAG eval gate

## Preconditions
1. Ollama up: `http://localhost:11434/api/tags` (model `llama3.2` or `LLM_MODEL`).
2. Index loaded (~195 chunks in demo): `data/vectorstore_faiss/` exists or reindex first.
3. Use project venv: `.\venv\Scripts\python.exe`.

## Workflow
```text
Eval progress:
- [ ] 1. Confirm Ollama
- [ ] 2. Run --compare (or plain eval)
- [ ] 3. Read RESULTADOS and regresiones
- [ ] 4. If intentional improvement and 12/12: --baseline
- [ ] 5. Do not commit .env / PDFs / vectorstore
```

### Commands (PowerShell, repo root)
```powershell
.\venv\Scripts\python.exe -u tests\eval_rag.py --compare
# Solo tras 12/12 y cambio acordado:
.\venv\Scripts\python.exe -u tests\eval_rag.py --baseline
```

`--compare` tarda ~10–15 min en CPU. Preferir `-u` / `PYTHONUNBUFFERED=1`.

## Interpreting results
- **PASS/total** = accuracy del harness (no es P@k IR).
- Solder paste: mirar `precision_paginas` / `recall_paginas` de la nota 📄.
- Regresión = un caso que era PASS en `tests/eval_baseline.json` y ahora FAIL.
- Casos OOS (paella/caballo): deben devolver `RESPUESTA_FUERA_DE_AMBITO`.
- Aspect Ratio: no debe aparecer `0.66` si la query no pide Area Ratio.
- Area Ratio (ES): debe poder satisfacer `apertura` / `paredes` (glosa o LLM).

## After failures
1. Identificar si el fallo es retrieval, cita literal, LLM o postproceso.
2. Evitar parches hardcodeados por documento; preferir scoring genérico o filtrado por término.
3. Re-ejecutar `--compare` antes de baseline/commit.

## Related
- Estándares del repo: rule `rag-standards`
- Feature workflow: skill `rag-feature`
