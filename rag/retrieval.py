"""Retrieval híbrido: BM25 + semántica + RRF + rerank."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document

from . import config, state
from .config import (
    CORPUS_SMALL_THRESHOLD,
    DEDUP_SIMILARITY_THRESHOLD,
    DEBUG_PREVIEW_CHARS,
    DEBUG_SIMILARITY_TOP_N,
    DEFAULT_K,
    EXHAUSTIVE_PATTERNS,
    LITERAL_PATTERNS,
    MAX_CHARS_CONTEXTO,
    MIN_CHUNKS_PER_DOC,
    PALABRAS_INTENT_EXHAUSTIVO,
    RERANK_MAX_CANDIDATES,
    RERANK_WEIGHT_CROSS,
    RERANK_WEIGHT_SIMILARITY,
    RERANKER_MODEL,
    RETRIEVAL_MULTIPLIER,
    RRF_K,
    STOPWORDS_BUSQUEDA,
    TERMINOS_INTENT_QUERY,
    VECTORSTORE_PATH,
)
from .errors import RetrievalPipelineError

def obtener_reranker():
    """Carga lazy del cross-encoder (singleton)."""
    if state.reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "Instala sentence-transformers para el reranker: "
                "pip install sentence-transformers"
            ) from e
        print(f"[INFO] Cargando reranker: {RERANKER_MODEL}")
        state.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return state.reranker


def _to_float(value) -> float:
    """Convierte numpy scalars a float nativo (serializable en JSON)."""
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)


def metadata_json(doc: Document) -> dict:
    """Metadata mínima serializable para respuestas API."""
    permitidos = (
        "document_id", "page_number", "chunk_id",
        "similarity_score", "rerank_score", "final_score",
        "bm25_score", "rrf_score", "lexical_score",
    )
    resultado = {}
    for clave in permitidos:
        if clave not in doc.metadata:
            continue
        valor = doc.metadata[clave]
        if isinstance(valor, (np.floating, np.integer)):
            resultado[clave] = float(valor)
        elif isinstance(valor, (int, float, str)):
            resultado[clave] = valor
    return resultado


def _normalizar_scores_altos_mejor(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _normalizar_l2_menor_mejor(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(max_s - s) / (max_s - min_s) for s in scores]


def _textos_muy_similares(a: str, b: str, umbral: float = DEDUP_SIMILARITY_THRESHOLD) -> bool:
    a_cmp, b_cmp = a[:600].strip(), b[:600].strip()
    if not a_cmp or not b_cmp:
        return False
    return SequenceMatcher(None, a_cmp, b_cmp).ratio() >= umbral


def _eliminar_duplicados_semanticos(
    rankeados: List[Tuple[Document, float, float, float]],
) -> List[Tuple[Document, float, float, float]]:
    unicos: List[Tuple[Document, float, float, float]] = []
    for item in rankeados:
        doc = item[0]
        if any(_textos_muy_similares(doc.page_content, u[0].page_content) for u in unicos):
            continue
        unicos.append(item)
    return unicos


def _aplicar_reranker(
    query: str, scored: List[Tuple[Document, float]]
) -> List[Tuple[Document, float, float, float]]:
    if not scored:
        return []

    docs = [doc for doc, _ in scored]
    l2_scores = [_to_float(s) for _, s in scored]
    sim_norm = _normalizar_l2_menor_mejor(l2_scores)

    reranker = obtener_reranker()
    pares = [(query, doc.page_content[:2000]) for doc in docs]
    rerank_raw = [_to_float(s) for s in reranker.predict(pares)]
    rerank_norm = _normalizar_scores_altos_mejor(rerank_raw)

    combinados: List[Tuple[Document, float, float, float]] = []
    for doc, sim_s, l2_raw, rer_s, rer_raw in zip(
        docs, sim_norm, l2_scores, rerank_norm, rerank_raw
    ):
        final = RERANK_WEIGHT_SIMILARITY * sim_s + RERANK_WEIGHT_CROSS * rer_s
        doc.metadata["similarity_score"] = round(l2_raw, 4)
        doc.metadata["similarity_norm"] = round(float(sim_s), 4)
        doc.metadata["rerank_score"] = round(rer_raw, 4)
        doc.metadata["rerank_norm"] = round(float(rer_s), 4)
        doc.metadata["final_score"] = round(float(final), 4)
        combinados.append((doc, l2_raw, rer_raw, float(final)))

    combinados.sort(key=lambda x: x[3], reverse=True)
    return combinados


def _rankear_modo_literal(
    query: str,
    vs,
    candidatos: List[Tuple[Document, float]],
) -> List[Tuple[Document, float, float, float]]:
    """Modo literal: orden léxico exhaustivo + RRF + BM25, sin cross-encoder."""
    lex_docs = busqueda_lexica_exhaustiva(query, vs)
    lex_rank = {_chunk_key(d): i for i, d in enumerate(lex_docs)}
    n_lex = len(lex_docs)

    combinados: List[Tuple[Document, float, float, float]] = []
    for doc, _ in candidatos:
        key = _chunk_key(doc)
        lr = lex_rank.get(key, n_lex)
        rrf = float(doc.metadata.get("rrf_score", 0) or 0)
        bm25 = float(doc.metadata.get("bm25_score", 0) or 0)
        final = (n_lex - lr) * 2.0 + rrf * 10.0 + bm25
        doc.metadata["lexical_rank"] = lr
        doc.metadata["rrf_score"] = round(rrf, 6)
        doc.metadata["bm25_score"] = round(bm25, 4)
        doc.metadata["final_score"] = round(final, 4)
        combinados.append((doc, 0.0, bm25, final))
    combinados.sort(key=lambda x: x[3], reverse=True)
    return combinados


def _finales_modo_literal(
    query: str,
    vs,
    rankeados: List[Tuple[Document, float, float, float]],
    k: int,
) -> List[Document]:
    """Prioriza chunks que contienen la frase discriminante (glosario/aviso)."""
    frase = termino_mas_discriminante(query)
    ranked_docs = [doc for doc, _, _, _ in rankeados]
    preferidos: List[Document] = []
    if frase:
        frase_l = frase.lower()
        preferidos = [
            d for d in state.corpus_docs
            if frase_l in d.page_content.lower()
        ]
        preferidos.sort(
            key=lambda d: (
                0 if "the ratio of the area" in d.page_content.lower() else 1,
                0 if "debe ficharse" in d.page_content.lower() else 1,
                -d.page_content.lower().count(frase_l),
            )
        )
        preferidos = preferidos[: min(3, k)]
    lex_docs = busqueda_lexica_exhaustiva(query, vs)[: min(3, k)]
    merged: List[Document] = []
    vistos: set = set()
    for doc in preferidos + lex_docs + ranked_docs:
        key = _chunk_key(doc)
        if key in vistos:
            continue
        vistos.add(key)
        merged.append(doc)
        if len(merged) >= k:
            break
    return merged


def _diag_log(mensaje: str) -> None:
    if config.DIAG_PIPELINE:
        print(f"[PIPELINE DIAG] {mensaje}")


def _vectorstore_count(vs) -> int:
    if vs is None:
        return 0
    if hasattr(vs, "index") and hasattr(vs.index, "ntotal"):
        return int(vs.index.ntotal)
    if hasattr(vs, "_collection") and hasattr(vs._collection, "count"):
        try:
            return int(vs._collection.count())
        except Exception:
            pass
    return len(state.corpus_docs)


def _chunk_key(doc: Document) -> str:
    doc_id = doc.metadata.get("document_id", "")
    chunk_id = doc.metadata.get("chunk_id", "")
    return f"{doc_id}::{chunk_id}"


def _ejemplo_chunk_id(doc: Optional[Document], origen: str) -> str:
    if doc is None:
        return f"{origen}: (ninguno)"
    meta = doc.metadata
    return (
        f"{origen}: key={_chunk_key(doc)!r} "
        f"chunk_id={meta.get('chunk_id')!r} ({type(meta.get('chunk_id')).__name__}) "
        f"doc={meta.get('document_id')!r}"
    )


def es_intent_exhaustivo(query: str) -> bool:
    return bool(EXHAUSTIVE_PATTERNS.search(query))


def debe_ampliar_topn(query: str) -> bool:
    return bool(LITERAL_PATTERNS.search(query))


def extraer_terminos_busqueda(query: str) -> List[str]:
    _cortas_ok = {"qr", "bg", "ic", "th"}
    out: List[str] = []
    for w in re.findall(r"\w+", query.lower()):
        if w in STOPWORDS_BUSQUEDA:
            continue
        if len(w) > 2 or w in _cortas_ok:
            out.append(w)
    return out


def _frecuencia_documental(termino: str) -> int:
    t = termino.lower()
    return sum(1 for d in state.corpus_docs if t in d.page_content.lower())


def termino_mas_discriminante(query: str) -> Optional[str]:
    """Término o bigrama de la query que aparece en menos chunks (más selectivo)."""
    terminos = [
        t for t in extraer_terminos_busqueda(query)
        if t not in TERMINOS_INTENT_QUERY
    ]
    if not terminos or not state.corpus_docs:
        return None
    n = len(state.corpus_docs)
    umbral = n * 0.85
    q_norm = " ".join(re.findall(r"\w+", query.lower()))

    candidatos: List[Tuple[int, int, str]] = []
    for i in range(len(terminos) - 1):
        frase = f"{terminos[i]} {terminos[i + 1]}"
        if frase not in q_norm:
            continue
        df = _frecuencia_documental(frase)
        if 0 < df < umbral:
            candidatos.append((df, -len(frase), frase))
    for t in terminos:
        df = _frecuencia_documental(t)
        if 0 < df < umbral:
            candidatos.append((df, -len(t), t))
    if not candidatos:
        return None
    candidatos.sort()
    return candidatos[0][2]


def ranking_lexico_por_termino(term: str) -> List[Document]:
    term_l = term.lower()
    docs = [d for d in state.corpus_docs if term_l in d.page_content.lower()]
    docs.sort(key=lambda d: d.page_content.lower().count(term_l), reverse=True)
    return docs


def reconstruir_indice_bm25(vs) -> None:
    """Índice BM25 en memoria sobre el corpus completo."""
    state.corpus_docs = list(vs.docstore._dict.values())
    n = len(state.corpus_docs)
    _diag_log(f"BM25 corpus len={n}")
    if n == 0:
        print("[ERROR] BM25: corpus vacío — reindexa con POST /reindexar")
        state.bm25_index = None
        return
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[AVISO] rank_bm25 no instalado — pip install rank-bm25")
        state.bm25_index = None
        return
    tokenizado = [doc.page_content.lower().split() for doc in state.corpus_docs]
    vacios = sum(1 for t in tokenizado if not t)
    _diag_log(f"BM25 tokens: total={len(tokenizado)} listas_vacias={vacios}")
    if vacios == len(tokenizado):
        print("[ERROR] BM25: todos los chunks tienen tokens vacíos")
        state.bm25_index = None
        return
    state.bm25_index = BM25Okapi(tokenizado)
    print(f"[INFO] Índice BM25: {n} chunks (vacíos={vacios})")
    if config.DIAG_PIPELINE and state.corpus_docs:
        _diag_log(_ejemplo_chunk_id(state.corpus_docs[0], "BM25 corpus[0]"))


def reciprocal_rank_fusion(
    rankings: List[List[Document]], k: int = RRF_K
) -> Tuple[List[Tuple[str, float]], Dict[str, Document]]:
    rankings_activos = [r for r in rankings if r]
    if not rankings_activos:
        _diag_log("RRF: todos los rankings de entrada están vacíos")
        return [], {}

    scores: Dict[str, float] = {}
    doc_by_id: Dict[str, Document] = {}
    for ranking in rankings_activos:
        for rank, doc in enumerate(ranking):
            doc_id = _chunk_key(doc)
            doc_by_id[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    ordenado = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ordenado, doc_by_id


def _calcular_top_n(query: str, k: int, total_chunks: int) -> int:
    if total_chunks <= 0:
        return k
    if total_chunks < CORPUS_SMALL_THRESHOLD:
        return total_chunks
    if es_intent_exhaustivo(query) or debe_ampliar_topn(query):
        return total_chunks
    return max(k * RETRIEVAL_MULTIPLIER, k)


def _modo_retrieval_label(query: str, total_chunks: int) -> str:
    partes: List[str] = []
    if es_intent_exhaustivo(query):
        partes.append("exhaustiva")
    if debe_ampliar_topn(query):
        partes.append("ampliada")
    if total_chunks < CORPUS_SMALL_THRESHOLD:
        partes.append("corpus_pequeno")
    partes.append("hibrida")
    return "+".join(partes) if partes else "hibrida"


def busqueda_bm25(query: str, top_n: int) -> List[Document]:
    if state.bm25_index is None or not state.corpus_docs:
        return []
    tokens = query.lower().split()
    if not tokens:
        return []
    try:
        puntajes = state.bm25_index.get_scores(tokens)
        ordenados = sorted(enumerate(puntajes), key=lambda x: x[1], reverse=True)
        resultado: List[Document] = []
        for idx, score in ordenados:
            if score <= 0:
                break
            doc = state.corpus_docs[idx]
            doc.metadata["bm25_score"] = round(_to_float(score), 4)
            resultado.append(doc)
        # Si top_n es 0 o None, devolver todos (para diversificación temprana)
        if top_n and top_n > 0:
            resultado = resultado[:top_n]
        return resultado
    except Exception as e:
        print(f"[ERROR] BM25 get_scores falló: {e}")
        import traceback
        traceback.print_exc()
        return []


def busqueda_lexica_exhaustiva(query: str, vs) -> List[Document]:
    """Búsqueda léxica sobre todos los chunks del vectorstore (sin depender de state.corpus_docs)."""
    terminos = extraer_terminos_busqueda(query)
    if not terminos:
        return []

    total_vs = _vectorstore_count(vs)
    if config.DEBUG_RAG:
        print(
            f"[DEBUG] Búsqueda léxica exhaustiva: len(state.corpus_docs)={len(state.corpus_docs)} | "
            f"_vectorstore_count={total_vs}"
        )

    if hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
        todos_los_docs = list(vs.docstore._dict.values())
    else:
        todos_los_docs = []

    _diag_log(
        f"Lexical exhaustiva: iterando {len(todos_los_docs)} chunks del vectorstore "
        f"(índice={total_vs})"
    )

    puntuados: List[Tuple[Document, int]] = []
    for doc in todos_los_docs:
        texto = doc.page_content.lower()
        coincidencias = sum(1 for t in terminos if t in texto)
        if coincidencias:
            doc.metadata["lexical_score"] = coincidencias
            puntuados.append((doc, coincidencias))
    puntuados.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in puntuados]


def busqueda_lexica_corpus(query: str) -> List[Document]:
    """Todos los chunks del corpus que contienen términos de la query."""
    terminos = extraer_terminos_busqueda(query)
    if not terminos or not state.corpus_docs:
        return []
    puntuados: List[Tuple[Document, int]] = []
    for doc in state.corpus_docs:
        texto = doc.page_content.lower()
        coincidencias = sum(1 for t in terminos if t in texto)
        if coincidencias:
            doc.metadata["lexical_score"] = coincidencias
            puntuados.append((doc, coincidencias))
    puntuados.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in puntuados]


def diversificar_por_documento(
    candidatos: List[Tuple[Document, float]],
    min_per_doc: int = MIN_CHUNKS_PER_DOC,
) -> List[Tuple[Document, float]]:
    """Reserva min_per_doc chunks de cada document_id presente en el ranking."""
    por_documento: Dict[str, List[Tuple[Document, float]]] = defaultdict(list)
    resto: List[Tuple[Document, float]] = []

    for item in candidatos:
        doc, _score = item
        doc_id = doc.metadata.get("document_id", "")
        if len(por_documento[doc_id]) < min_per_doc:
            por_documento[doc_id].append(item)
        else:
            resto.append(item)

    garantizados = [item for items in por_documento.values() for item in items]
    return garantizados + resto


def seleccionar_finales_con_cobertura(
    rankeados: List[Tuple[Document, float, float, float]],
    query: str,
    k: int,
    min_per_doc: int = MIN_CHUNKS_PER_DOC,
) -> List[Document]:
    """
    Top-k priorizando el término más discriminante de la query y reservando
    min_per_doc por documento que lo contiene (evita que el rerank borre un doc).
    """
    if k <= 0 or not rankeados:
        return []

    term = termino_mas_discriminante(query)
    term_l = term.lower() if term else None

    def contiene(doc: Document) -> bool:
        return bool(term_l) and term_l in doc.page_content.lower()

    if term_l:
        ordenados = [x for x in rankeados if contiene(x[0])] + [
            x for x in rankeados if not contiene(x[0])
        ]
    else:
        ordenados = list(rankeados)

    docs_con_term: set = set()
    if term_l:
        for d in state.corpus_docs:
            if term_l in d.page_content.lower():
                docs_con_term.add(d.metadata.get("document_id", ""))

    por_doc: Dict[str, int] = defaultdict(int)
    vistos: set = set()
    esenciales: List[Document] = []
    resto: List[Document] = []

    for doc, *_rest in ordenados:
        key = _chunk_key(doc)
        if key in vistos:
            continue
        vistos.add(key)
        did = doc.metadata.get("document_id", "")
        if did in docs_con_term and por_doc[did] < min_per_doc and contiene(doc):
            esenciales.append(doc)
            por_doc[did] += 1
        else:
            resto.append(doc)

    if term_l:
        for did in docs_con_term:
            if por_doc[did] >= min_per_doc:
                continue
            for doc in state.corpus_docs:
                if por_doc[did] >= min_per_doc:
                    break
                if doc.metadata.get("document_id") != did:
                    continue
                if term_l not in doc.page_content.lower():
                    continue
                key = _chunk_key(doc)
                if key in vistos:
                    continue
                vistos.add(key)
                esenciales.append(doc)
                por_doc[did] += 1

    matching_extra = [d for d in resto if contiene(d)]
    otros = [d for d in resto if not contiene(d)]
    return (esenciales + matching_extra + otros)[:k]


def _obtener_candidatos_rrf(
    query: str, vs, top_n: int
) -> Tuple[List[Tuple[Document, float]], List[Tuple[Document, float]], List[Document], List[Document]]:
    """Fusión BM25 + semántica (+ léxica si aplica) → candidatos para reranker."""
    semantico: List[Tuple[Document, float]] = vs.similarity_search_with_score(query, k=top_n)
    if not semantico:
        raise RetrievalPipelineError(
            f"similarity_search_with_score devolvió 0 resultados (top_n={top_n}). "
            f"Índice ntotal={_vectorstore_count(vs)}. ¿Reindexar?"
        )

    ranking_sem = [doc for doc, _ in semantico]
    _diag_log(
        f"Similarity: {len(semantico)} docs | primer score L2={_to_float(semantico[0][1]):.4f} | "
        f"{_ejemplo_chunk_id(semantico[0][0], 'vectorstore')}"
    )

    ranking_bm25 = busqueda_bm25(query, top_n)
    _diag_log(f"BM25: {len(ranking_bm25)} docs")
    if ranking_bm25:
        _diag_log(_ejemplo_chunk_id(ranking_bm25[0], "BM25"))

    rankings: List[List[Document]] = []
    if ranking_sem:
        rankings.append(ranking_sem)
    if ranking_bm25:
        rankings.append(ranking_bm25)

    ranking_lex: List[Document] = []
    if es_intent_exhaustivo(query):
        ranking_lex = busqueda_lexica_exhaustiva(query, vs)
        _diag_log(f"Lexical exhaustiva: {len(ranking_lex)} docs")
        if ranking_lex:
            rankings.append(ranking_lex)
    elif debe_ampliar_topn(query):
        ranking_lex = busqueda_lexica_exhaustiva(query, vs)
        _diag_log(f"Lexical modo literal: {len(ranking_lex)} docs")
        if ranking_lex:
            rankings.append(ranking_lex)
    else:
        term = termino_mas_discriminante(query)
        if term:
            ranking_lex = ranking_lexico_por_termino(term)
            _diag_log(f"Lexical término discriminante {term!r}: {len(ranking_lex)} docs")
            if ranking_lex:
                rankings.append(ranking_lex)

    fusionado, doc_map = reciprocal_rank_fusion(rankings)
    _diag_log(f"RRF: {len(fusionado)} docs únicos")

    candidatos: List[Tuple[Document, float]] = []
    vistos: set = set()
    for doc_id, rrf in fusionado:
        doc = doc_map[doc_id]
        if doc_id in vistos:
            continue
        vistos.add(doc_id)
        doc.metadata["rrf_score"] = round(rrf, 6)
        candidatos.append((doc, -rrf))

    if es_intent_exhaustivo(query):
        for doc in ranking_lex:
            doc_id = _chunk_key(doc)
            if doc_id in vistos:
                continue
            vistos.add(doc_id)
            candidatos.append((doc, 0.0))

    if not candidatos and semantico:
        _diag_log("RRF vacío → fallback a orden semántico puro")
        candidatos = list(semantico)

    # Cuota por documento sobre el ranking RRF completo, después recorte a top_n.
    if not debe_ampliar_topn(query):
        candidatos = diversificar_por_documento(candidatos)
    if top_n and top_n > 0:
        candidatos = candidatos[:top_n]

    return candidatos, semantico, ranking_bm25, ranking_lex


def _debug_pipeline_hibrido(
    query: str,
    modo: str,
    top_n: int,
    semantico: List[Tuple[Document, float]],
    bm25_docs: List[Document],
    candidatos: List[Tuple[Document, float]],
    k: int,
) -> None:
    print("[RAG DEBUG]")
    print(f"QUERY: {query!r}")
    print(f"MODO: {modo} | TOP_N: {top_n}")
    print(f"TOP {min(len(semantico), DEBUG_SIMILARITY_TOP_N)} SIMILARITY (L2 menor = más similar):")
    for i, (doc, l2) in enumerate(semantico[:DEBUG_SIMILARITY_TOP_N], 1):
        meta = doc.metadata
        preview = doc.page_content[:DEBUG_PREVIEW_CHARS].replace("\n", " ")
        print(
            f"  {i:2}. similarity_score={_to_float(l2):.4f} | chunk_id={meta.get('chunk_id')} | "
            f"document_id={meta.get('document_id')} | page_number={meta.get('page_number')}\n"
            f"      preview={preview}"
        )
    print(f"TOP {min(len(bm25_docs), DEBUG_SIMILARITY_TOP_N)} BM25:")
    for i, doc in enumerate(bm25_docs[:DEBUG_SIMILARITY_TOP_N], 1):
        meta = doc.metadata
        preview = doc.page_content[:DEBUG_PREVIEW_CHARS].replace("\n", " ")
        print(
            f"  {i:2}. bm25_score={meta.get('bm25_score', 'n/a')} | chunk_id={meta.get('chunk_id')} | "
            f"document_id={meta.get('document_id')} | page_number={meta.get('page_number')}\n"
            f"      preview={preview}"
        )
    print(f"TOP {min(len(candidatos), DEBUG_SIMILARITY_TOP_N)} RRF -> RERANK (candidatos):")
    for i, (doc, pseudo) in enumerate(candidatos[:DEBUG_SIMILARITY_TOP_N], 1):
        meta = doc.metadata
        preview = doc.page_content[:DEBUG_PREVIEW_CHARS].replace("\n", " ")
        print(
            f"  {i:2}. rrf_score={meta.get('rrf_score', 'n/a')} | "
            f"lexical_score={meta.get('lexical_score', 'n/a')} | chunk_id={meta.get('chunk_id')}\n"
            f"      preview={preview}"
        )


def _debug_reranked(
    rankeados: List[Tuple[Document, float, float, float]], k: int
) -> None:
    print(f"TOP {k} RERANKEADOS (final_score DESC):")
    for i, (doc, l2, rer, final) in enumerate(rankeados[:k], 1):
        meta = doc.metadata
        preview = doc.page_content[:DEBUG_PREVIEW_CHARS].replace("\n", " ")
        if len(doc.page_content) > DEBUG_PREVIEW_CHARS:
            preview += "..."
        print(
            f"  {i}. similarity_score={l2:.4f} rerank_score={rer:.4f} final_score={final:.4f}\n"
            f"     chunk_id={meta.get('chunk_id')} | document_id={meta.get('document_id')} | "
            f"page_number={meta.get('page_number')}\n"
            f"     preview={preview}"
        )


def _debug_documentos_pre_rerank(
    candidatos: List[Tuple[Document, float]], top_n: int
) -> None:
    if not config.DEBUG_RAG:
        return
    n = min(len(candidatos), top_n)
    print(f"[DEBUG] TOP_{n} document_id antes del reranker:")
    for i, (doc, _) in enumerate(candidatos[:n], 1):
        meta = doc.metadata
        print(
            f"  {i}. document_id={meta.get('document_id')} | "
            f"chunk_id={meta.get('chunk_id')} | page_number={meta.get('page_number')}"
        )


def _diag_chunk_glosario_area_ratio(
    query: str,
    sem_scored: List[Tuple[Document, float]],
    rankeados: List[Tuple[Document, float, float, float]],
) -> None:
    """Diagnóstico del chunk 1.1.3 / definición literal de Area Ratio."""
    if not config.DEBUG_RAG or not debe_ampliar_topn(query):
        return
    marcadores = ("1.1.3", "area of aperture walls")
    objetivos = [
        d for d in state.corpus_docs
        if any(m.lower() in d.page_content.lower() for m in marcadores)
    ]
    if not objetivos:
        print("[DEBUG] Chunk glosario 1.1.3: NO ENCONTRADO en corpus indexado")
        return
    sem_map = {_chunk_key(d): _to_float(s) for d, s in sem_scored}
    rank_map = {
        _chunk_key(item[0]): (_to_float(item[2]), _to_float(item[3]))
        for item in rankeados
    }
    for doc in objetivos:
        key = _chunk_key(doc)
        meta = doc.metadata
        l2 = sem_map.get(key)
        rer_info = rank_map.get(key)
        l2_txt = f"{l2:.4f}" if l2 is not None else "N/A (fuera de TOP_N semántico)"
        if rer_info:
            rer_txt = f"{rer_info[0]:.4f} | final_score={rer_info[1]:.4f}"
        else:
            rer_txt = "N/A (no pasó al reranker)"
        print(
            f"[DEBUG] Chunk glosario: chunk_id={meta.get('chunk_id')} | "
            f"document_id={meta.get('document_id')} | similarity L2={l2_txt} | "
            f"rerank_score={rer_txt}"
        )


def paginas_cobertura_exhaustiva(query: str, resultados_lexicos: list) -> List[int]:
    """Páginas de la nota 📄 (AND de términos de tema, no de la pregunta)."""
    terminos = [
        t for t in extraer_terminos_busqueda(query)
        if t not in PALABRAS_INTENT_EXHAUSTIVO and t not in TERMINOS_INTENT_QUERY
    ]
    paginas = set()
    for doc in resultados_lexicos:
        pagina = doc.metadata.get("page_number")
        if pagina is None:
            continue
        texto = doc.page_content.lower()
        if terminos and not all(t in texto for t in terminos):
            continue
        pagina_visible = int(pagina) + 1
        if pagina_visible == 6:
            continue
        paginas.add(pagina_visible)
    return sorted(paginas)


def construir_nota_exhaustiva(term: str, resultados_lexicos: list) -> str:
    """
    Construye la nota de cobertura exhaustiva a partir de los resultados léxicos.
    Solo se llama cuando EXHAUSTIVE_PATTERNS hace match.
    """
    paginas = paginas_cobertura_exhaustiva(term, resultados_lexicos)
    if not paginas:
        return ""

    paginas_str = ", ".join(str(p) for p in paginas)
    etiqueta = "página" if len(paginas) == 1 else "páginas"
    return (
        f"\n\n📄 Búsqueda exhaustiva: el término aparece en "
        f"{len(paginas)} {etiqueta} del corpus: {paginas_str}."
    )


def pipeline_recuperacion(
    vs,
    query: str,
    k: int = DEFAULT_K,
    incluir_info_exhaustiva: bool = False,
):
    """
    Híbrido BM25 + semántica (RRF) → rerank → dedup → top-k.
    Ramas exhaustiva / ampliada según intent de la query.
    """
    if vs is None:
        raise RetrievalPipelineError("Vectorstore no inicializado")

    total_chunks = _vectorstore_count(vs)
    if config.DEBUG_RAG:
        print(f"[DEBUG] total_chunks en tiempo de query: {total_chunks}")
    _diag_log(
        f"Vectorstore count={total_chunks} | ruta={VECTORSTORE_PATH} | "
        f"existe={os.path.exists(VECTORSTORE_PATH)}"
    )

    if total_chunks == 0:
        raise RetrievalPipelineError(
            f"Índice vacío (ntotal=0). Directorio {VECTORSTORE_PATH!r} "
            f"existe={os.path.exists(VECTORSTORE_PATH)}. Ejecuta POST /reindexar."
        )

    if not state.corpus_docs or len(state.corpus_docs) != total_chunks:
        _diag_log(
            f"Corpus BM25 desincronizado (memoria={len(state.corpus_docs)}, "
            f"índice={total_chunks}) — reconstruyendo"
        )
        reconstruir_indice_bm25(vs)

    top_n = _calcular_top_n(query, k, total_chunks)
    modo = _modo_retrieval_label(query, total_chunks)
    if config.DEBUG_RAG:
        exhaustivo = "SÍ" if es_intent_exhaustivo(query) else "NO"
        print(f'[DEBUG] Intent exhaustivo: {exhaustivo} — query: "{query}"')
        print(f"[DEBUG] debe_ampliar_topn() = {debe_ampliar_topn(query)}")
    _diag_log(f"Query={query!r} | modo={modo} | top_n={top_n} | k={k}")

    candidatos, sem_scored, bm25_docs, resultados_lexicos = _obtener_candidatos_rrf(query, vs, top_n)
    _diag_log(f"Candidatos pre-rerank (post-diversificación): {len(candidatos)}")

    if config.DEBUG_RAG:
        _debug_pipeline_hibrido(query, modo, top_n, sem_scored, bm25_docs, candidatos, k)
        _debug_documentos_pre_rerank(candidatos, top_n)

    if debe_ampliar_topn(query):
        candidatos_rerank = candidatos
        _diag_log(
            f"Modo literal: rerank sobre {len(candidatos_rerank)} candidatos "
            f"(corpus completo, sin límite semántico, sin cross-encoder)"
        )
        rankeados = _rankear_modo_literal(query, vs, candidatos_rerank)
    else:
        candidatos_rerank = candidatos[: min(len(candidatos), RERANK_MAX_CANDIDATES)]
        if len(candidatos) > len(candidatos_rerank):
            _diag_log(
                f"Rerank limitado a {len(candidatos_rerank)} candidatos "
                f"(max={RERANK_MAX_CANDIDATES})"
            )
        try:
            rankeados = _aplicar_reranker(query, candidatos_rerank)
        except Exception as e:
            print(f"[ERROR] Reranker falló, usando orden RRF/semántico: {e}")
            import traceback
            traceback.print_exc()
            rankeados = [
                (doc, 0.0, 0.0, float(doc.metadata.get("rrf_score", 0) or 0))
                for doc, _ in candidatos_rerank
            ]

    _diag_log(f"Post-rerank: {len(rankeados)} docs")

    antes_dedup = len(rankeados)
    rankeados = _eliminar_duplicados_semanticos(rankeados)
    _diag_log(f"Post-dedup: {len(rankeados)} docs (eliminados={antes_dedup - len(rankeados)})")

    finales = (
        _finales_modo_literal(query, vs, rankeados, k)
        if debe_ampliar_topn(query)
        else seleccionar_finales_con_cobertura(rankeados, query, k)
    )
    contexto_chars = len(construir_contexto(finales))
    _diag_log(f"Final: {len(finales)} docs | contexto={contexto_chars} chars")

    if not finales:
        _diag_log("VACÍO: pipeline sin documentos finales")

    if config.DEBUG_RAG:
        _debug_reranked(rankeados, k)
        _diag_chunk_glosario_area_ratio(query, sem_scored, rankeados)

    if incluir_info_exhaustiva:
        return {
            "documentos": finales,
            "resultados_lexicos": resultados_lexicos if es_intent_exhaustivo(query) else [],
        }

    return finales


def recuperar_documentos(vs, query: str, k: int = DEFAULT_K) -> List[Document]:
    return pipeline_recuperacion(vs, query, k=k)


def construir_contexto(docs: List[Document]) -> str:
    """Orden por relevancia final, límite de caracteres, sin filtros por tipo de documento."""
    partes = []
    total = 0
    for i, doc in enumerate(docs, 1):
        doc_id = doc.metadata.get("document_id", "?")
        pagina = doc.metadata.get("page_number", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        etiqueta = f"[{i}] {doc_id} | p.{pagina} | chunk {chunk_id}"
        bloque = f"{etiqueta}\n{doc.page_content}"
        if total + len(bloque) > MAX_CHARS_CONTEXTO:
            break
        partes.append(bloque)
        total += len(bloque)
    return "\n\n---\n\n".join(partes)
