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

## Estado de evaluación: 8/10 PASS (baseline 30/06/2026)

Fuente de verdad: `eval_baseline.json`. Vacaciones y QR **ya pasan**. El 7/10 de notas antiguas está desfasado.

### Tests que funcionan
1. ¿Qué es el Area Ratio?
2. Copia literal definición Area Ratio
3. ¿En qué páginas aparece solder paste?
4. ¿Cómo se solicitan las vacaciones?
5. ¿Cómo se ficha la entrada y la salida por QR?
6. ¿Qué dice exactamente sobre compartir credenciales?
7. Copia literal aviso fichaje QR
8. Resume descansos y pausas

### Tests que fallaban en el baseline

#### FAIL 1: Aspect Ratio
**Query**: "¿Qué recomendaciones hay sobre Aspect Ratio?"
**Error**: LLM dice `0.66` (umbral de Area Ratio). El correcto es `>1.5`.
**Fix aplicado**: priorizar el término más discriminante (`aspect`) en RRF y en el top-k final; regla de prompt para no mezclar umbrales de conceptos distintos.

#### FAIL 2: BGA
**Query**: "Resume los apartados relacionados con BGA."
**Error**: retrieval no incluye `IPC-7525A`.
**Causa**: `diversificar_por_documento` solo reordenaba candidatos que ya habían pasado el corte; el rerank + top-k=8 podía dejar fuera 7525A.
**Fix aplicado**: cuota por documento **antes** del corte `top_n` sobre el ranking RRF completo; cobertura del término discriminante (`bga`) al elegir el top-k final.

Tras estos cambios hay que reejecutar `python eval_rag.py` (y `--baseline` si 10/10 o si el nuevo 8+/10 debe ser la referencia).

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

**Generado**: 01/09/2026
**Objetivo**: confirmar 10/10 con `eval_rag.py` o decidir qué tests abandonar
