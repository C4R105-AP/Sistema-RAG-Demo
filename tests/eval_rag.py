"""
Evaluación de regresión del pipeline RAG.

Uso:
  python tests/eval_rag.py              # ejecutar tests y mostrar resultado
  python tests/eval_rag.py --baseline   # guardar resultado como baseline
  python tests/eval_rag.py --compare    # comparar con baseline; exit 1 si hay regresión

Reglas:
  1. Ejecutar --baseline antes de cambiar el pipeline.
  2. Aplicar cambio (+ reindexar si aplica).
  3. Ejecutar --compare; si un test PASS pasa a FAIL, revertir el cambio.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# Silenciar debug del pipeline durante eval (se puede sobreescribir con EVAL_DEBUG=1)
if os.getenv("EVAL_DEBUG", "").lower() not in ("1", "true", "yes"):
    os.environ["DEBUG_RAG"] = "false"
    os.environ["DIAG_PIPELINE"] = "false"

import rag.api as api_rag  # noqa: E402

BASELINE_PATH = Path(__file__).resolve().parent / "eval_baseline.json"

CASOS_TEST = [
    {
        "query": "¿Qué es el Area Ratio?",
        "debe_contener": ["apertura", "paredes"],
        "no_debe_decir": ["no tengo información"],
    },
    {
        "query": "Copia literalmente la definición de Area Ratio.",
        "debe_contener_exacto": (
            "The ratio of the area of aperture opening to the area of aperture walls"
        ),
    },
    {
        "query": "¿En qué páginas aparece el término solder paste?",
        "paginas_esperadas": [5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19],
        "tolerancia_min_paginas": 10,
        "debe_contener_nota_exhaustiva": True,
    },
    {
        "query": "¿Qué recomendaciones hay sobre Aspect Ratio?",
        "debe_contener": [">1.5", "aspect ratio"],
        "no_debe_contener": ["0.66"],
    },
    {
        "query": "Resume los apartados relacionados con BGA.",
        "debe_incluir_documento": "IPC-7525A",
    },
    # Tests de Bizneo
    {
        "query": "¿Cómo se solicitan las vacaciones?",
        "debe_contener": ["calendario", "ausencias", "vacaciones"],
        "no_debe_decir": ["no tengo información suficiente"],
    },
    {
        "query": "¿Cómo se ficha la entrada y la salida por QR?",
        "debe_contener": ["código QR", "iniciar", "finalizar"],
        "no_debe_decir": ["no tengo información suficiente"],
        "no_debe_repetir_parrafos": True,
    },
    {
        "query": "¿Qué dice exactamente el documento sobre compartir credenciales?",
        "debe_contener": ["credenciales", "compartir"],
        "no_debe_decir": ["no tengo información suficiente"],
    },
    {
        "query": "Copia literalmente el aviso importante del fichaje por QR.",
        "debe_contener_exacto": "Debe ficharse al iniciar y al finalizar la jornada",
    },
    {
        "query": "Resume todos los apartados relacionados con descansos y pausas.",
        "debe_contener": ["descanso", "pausa", "no deben contar"],
        "no_debe_decir": ["no tengo información suficiente"],
    },
]


def _texto_docs(docs: List[api_rag.Document]) -> str:
    return "\n".join(d.page_content for d in docs)


def _documentos_fuente(docs: List[api_rag.Document]) -> List[str]:
    return sorted(
        {
            str(d.metadata.get("document_id", ""))
            for d in docs
            if d.metadata.get("document_id")
        }
    )


def _paginas_de_docs(docs: List[api_rag.Document]) -> Set[int]:
    paginas: Set[int] = set()
    for d in docs:
        p = d.metadata.get("page_number")
        if p is None:
            continue
        try:
            paginas.add(int(p))
        except (TypeError, ValueError):
            pass
    return paginas


def _snapshot_config() -> Dict[str, Any]:
    return {
        "CHUNK_SIZE": api_rag.CHUNK_SIZE,
        "CHUNK_OVERLAP": api_rag.CHUNK_OVERLAP,
        "CHUNK_SIZE_env": os.getenv("CHUNK_SIZE"),
        "CHUNK_OVERLAP_env": os.getenv("CHUNK_OVERLAP"),
        "DEFAULT_K": api_rag.DEFAULT_K,
        "total_chunks": None,
    }


def _cargar_sistema() -> None:
    api_rag.DEBUG_RAG = os.getenv("EVAL_DEBUG", "").lower() in ("1", "true", "yes")
    api_rag.DIAG_PIPELINE = api_rag.DEBUG_RAG
    if not api_rag.cargar_vectorstore_global():
        raise RuntimeError("No se pudo cargar el vectorstore. ¿Existe vectorstore_faiss/?")
    if api_rag.vectorstore is None:
        raise RuntimeError("Vectorstore es None tras cargar_vectorstore_global()")


def validar_caso(caso: Dict[str, Any], eval_data: Dict[str, Any]) -> Dict[str, Any]:
    """Valida un caso según las claves presentes en el dict del test."""
    query = caso["query"]
    docs: List[api_rag.Document] = eval_data["docs"]
    contexto: str = eval_data["contexto"]
    respuesta_llm: str = eval_data.get("respuesta_llm", "")
    texto_retrieval = (contexto + "\n" + _texto_docs(docs)).lower()
    texto_llm = respuesta_llm.lower()

    resultado: Dict[str, Any] = {
        "query": query,
        "ok": True,
        "motivo": "",
        "docs_count": len(docs),
        "documentos_fuente": _documentos_fuente(docs),
        "chunk_ids": [d.metadata.get("chunk_id") for d in docs],
    }

    motivos: List[str] = []

    if "debe_contener" in caso:
        for termino in caso["debe_contener"]:
            t = termino.lower()
            if t not in texto_retrieval and t not in texto_llm:
                motivos.append(f"falta texto requerido: {termino!r}")

    if "debe_contener_exacto" in caso:
        exacto = caso["debe_contener_exacto"]
        if exacto not in contexto and exacto not in _texto_docs(docs):
            if exacto not in respuesta_llm:
                motivos.append(
                    f"falta cita exacta en retrieval/LLM: {exacto!r}"
                )

    if "no_debe_decir" in caso:
        for frase in caso["no_debe_decir"]:
            if frase.lower() in texto_llm:
                motivos.append(f"LLM dijo frase prohibida: {frase!r}")

    if "no_debe_contener" in caso:
        for frase in caso["no_debe_contener"]:
            if frase.lower() in texto_llm:
                motivos.append(f"LLM contiene texto prohibido: {frase!r}")

    if "debe_contener_nota_exhaustiva" in caso:
        # Verificar que la nota exhaustiva aparece con páginas reales (no solo el encabezado)
        tiene_nota = "📄 Búsqueda exhaustiva" in respuesta_llm or "📄 Busqueda exhaustiva" in respuesta_llm
        if not tiene_nota:
            motivos.append("falta nota exhaustiva (📄 Búsqueda exhaustiva) en respuesta del LLM")
        else:
            # Confirmar que hay números de página reales después del marcador
            fragmento_nota = respuesta_llm[respuesta_llm.find("📄"):] if "📄" in respuesta_llm else ""
            if fragmento_nota:
                numeros_encontrados = re.findall(r"\b\d+\b", fragmento_nota)
                if not numeros_encontrados:
                    motivos.append(
                        "nota exhaustiva presente pero sin números de página reales"
                    )

    if "paginas_esperadas" in caso:
        esperadas = set(caso["paginas_esperadas"])
        minimo = caso.get("tolerancia_min_paginas", len(esperadas))
        paginas_encontradas = eval_data.get("paginas_exhaustivas", set())
        coinciden = esperadas & paginas_encontradas
        resultado["paginas_encontradas"] = sorted(paginas_encontradas)
        resultado["paginas_coincidentes"] = sorted(coinciden)
        if len(coinciden) < minimo:
            motivos.append(
                f"páginas solder paste: {len(coinciden)}/{len(esperadas)} "
                f"(mínimo {minimo}). Encontradas: {sorted(coinciden)}"
            )

    if "debe_incluir_documento" in caso:
        doc_id_req = caso["debe_incluir_documento"]
        fuentes = resultado["documentos_fuente"]
        # Validar contra retrieval
        aparece_en_retrieval = any(doc_id_req in f for f in fuentes)
        # Validar contra la respuesta del LLM (detecta si el LLM ignoró el documento)
        aparece_en_llm = doc_id_req.lower() in respuesta_llm.lower()
        if not aparece_en_retrieval:
            motivos.append(
                f"retrieval no incluye documento {doc_id_req!r}. "
                f"Fuentes: {fuentes}"
            )
        elif not aparece_en_llm:
            motivos.append(
                f"documento {doc_id_req!r} presente en retrieval pero "
                f"IGNORADO por el LLM en la respuesta"
            )

    if "no_debe_repetir_parrafos" in caso and caso["no_debe_repetir_parrafos"]:
        # Detectar repetición de párrafos (>50 palabras)
        palabras = respuesta_llm.split()
        vistos: set = set()
        for i in range(len(palabras) - 50):
            bloque = " ".join(palabras[i:i+50])
            if bloque in vistos:
                motivos.append("respuesta contiene párrafos repetidos")
                break
            vistos.add(bloque)

    if motivos:
        resultado["ok"] = False
        resultado["motivo"] = "; ".join(motivos)

    return resultado


def ejecutar_evaluacion() -> Dict[str, Any]:
    _cargar_sistema()
    vs = api_rag.vectorstore
    assert vs is not None

    config = _snapshot_config()
    config["total_chunks"] = api_rag._vectorstore_count(vs)
    config["bm25_chunks"] = len(api_rag._corpus_docs)
    config["sincronizado"] = config["bm25_chunks"] == config["total_chunks"]

    qa = api_rag.crear_qa_chain(vs)
    resultados: List[Dict[str, Any]] = []

    for caso in CASOS_TEST:
        query = caso["query"]
        
        # Pipeline de recuperación (sin timeout - Ollama necesita el thread principal)
        print(f"  Recuperando: {query[:50]}...", flush=True)
        docs = api_rag.pipeline_recuperacion(vs, query, k=api_rag.DEFAULT_K)
        
        contexto = api_rag.construir_contexto(docs)

        paginas_exhaustivas: Set[int] = set()
        if "paginas_esperadas" in caso:
            lex_docs = api_rag.busqueda_lexica_exhaustiva(query, vs)
            paginas_exhaustivas = _paginas_de_docs(lex_docs)

        # Generación LLM (sin timeout - Ollama necesita el thread principal)
        print(f"  Generando respuesta LLM: {query[:50]}...", flush=True)
        try:
            qa_out = qa({"query": query, "k": api_rag.DEFAULT_K})
            respuesta_llm = qa_out.get("result", "")
        except Exception as e:
            print(f"[ERROR] LLM falló para query '{query}': {e}", flush=True)
            respuesta_llm = ""

        eval_data = {
            "docs": docs,
            "contexto": contexto,
            "respuesta_llm": respuesta_llm,
            "paginas_exhaustivas": paginas_exhaustivas,
        }
        resultados.append(validar_caso(caso, eval_data))

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "resultados": resultados,
        "pass": sum(1 for r in resultados if r["ok"]),
        "fail": sum(1 for r in resultados if not r["ok"]),
        "total": len(resultados),
    }


def guardar_baseline(reporte: Dict[str, Any]) -> None:
    BASELINE_PATH.write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Baseline guardado en {BASELINE_PATH}")


def cargar_baseline() -> Dict[str, Any]:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"No existe baseline en {BASELINE_PATH}. Ejecuta: python eval_rag.py --baseline"
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def comparar_con_baseline(actual: Dict[str, Any], baseline: Dict[str, Any]) -> List[str]:
    regresiones: List[str] = []
    base_por_query = {r["query"]: r for r in baseline["resultados"]}
    for r in actual["resultados"]:
        q = r["query"]
        prev = base_por_query.get(q)
        if prev is None:
            continue
        if prev.get("ok") and not r.get("ok"):
            regresiones.append(f"REGRESIÓN: {q!r} — antes PASS, ahora FAIL: {r.get('motivo')}")
    return regresiones


def imprimir_reporte(reporte: Dict[str, Any]) -> None:
    cfg = reporte["config"]
    print("=" * 60)
    print("CONFIGURACIÓN EN EVALUACIÓN")
    print("=" * 60)
    print(f"  CHUNK_SIZE (código):  {cfg['CHUNK_SIZE']}")
    print(f"  CHUNK_OVERLAP:        {cfg['CHUNK_OVERLAP']}")
    print(f"  CHUNK_SIZE (.env):    {cfg.get('CHUNK_SIZE_env')}")
    print(f"  CHUNK_OVERLAP (.env): {cfg.get('CHUNK_OVERLAP_env')}")
    print(f"  total_chunks:         {cfg['total_chunks']}")
    print(f"  bm25_chunks:          {cfg['bm25_chunks']}")
    print(f"  sincronizado:         {cfg['sincronizado']}")
    print(f"  timestamp:            {reporte['timestamp']}")
    print("=" * 60)
    print(f"RESULTADOS: {reporte['pass']}/{reporte['total']} PASS")
    print("=" * 60)
    for r in reporte["resultados"]:
        estado = "PASS" if r["ok"] else "FAIL"
        print(f"[{estado}] {r['query']}")
        if not r["ok"]:
            print(f"  Motivo: {r['motivo']}")
        if r.get("documentos_fuente"):
            print(f"  Fuentes: {r['documentos_fuente']}")
        if r.get("chunk_ids") is not None:
            print(f"  chunk_ids: {r['chunk_ids']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación de regresión RAG")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Guardar resultado actual como baseline",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Comparar con baseline y fallar si hay regresiones",
    )
    args = parser.parse_args()

    print("Cargando índice y ejecutando evaluación (puede tardar varios minutos)...")
    reporte = ejecutar_evaluacion()
    imprimir_reporte(reporte)

    if args.baseline:
        guardar_baseline(reporte)

    if args.compare:
        try:
            baseline = cargar_baseline()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 2
        print("=" * 60)
        print("COMPARACIÓN CON BASELINE")
        print(f"  Baseline: {baseline.get('timestamp')}")
        print(f"  Baseline chunks: {baseline.get('config', {}).get('total_chunks')}")
        regresiones = comparar_con_baseline(reporte, baseline)
        if regresiones:
            print("REGRESIONES DETECTADAS:")
            for msg in regresiones:
                print(f"  - {msg}")
            return 1
        print("Sin regresiones respecto al baseline.")
        mejoras = [
            r["query"]
            for r in reporte["resultados"]
            if r["ok"]
            and not next(
                (b["ok"] for b in baseline["resultados"] if b["query"] == r["query"]),
                False,
            )
        ]
        if mejoras:
            print(f"Mejoras (FAIL->PASS): {mejoras}")

    return 0 if reporte["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
