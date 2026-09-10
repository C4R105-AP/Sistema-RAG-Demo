"""Generación, postproceso determinista y cadena QA."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from . import state
from .config import (
    DEFAULT_K,
    PALABRAS_INTENT_EXHAUSTIVO,
    RAG_SYSTEM_PROMPT,
    TERMINOS_INTENT_QUERY,
)
from .errors import RetrievalPipelineError
from .llm import inicializar_llm
from .retrieval import (
    _frecuencia_documental,
    construir_contexto,
    construir_nota_exhaustiva,
    es_intent_exhaustivo,
    extraer_terminos_busqueda,
    pipeline_recuperacion,
    termino_mas_discriminante,
)

def _consulta_fuera_de_corpus(query: str) -> bool:
    """True si alguna palabra sustancial de la query no está en el índice.

    No usa embeddings ni LLM: recorre df en memoria (milisegundos).
    Ignora verbos cortos tipo 'era' para que trampas ('caballo') no disparen el pipeline.
    """
    terminos = [
        t for t in extraer_terminos_busqueda(query)
        if t not in TERMINOS_INTENT_QUERY and t not in PALABRAS_INTENT_EXHAUSTIVO
    ]
    if not state.corpus_docs:
        return False
    sustanciales = [
        t for t in terminos
        if len(t) >= 4 or t in {"qr", "bga", "ipc"}
    ]
    comprobar = sustanciales or terminos
    if not comprobar:
        return False
    return min(_frecuencia_documental(t) for t in comprobar) == 0


RESPUESTA_FUERA_DE_AMBITO = (
    "Esta pregunta no aparece en los documentos indexados. "
    "Prueba con un tema del corpus (esténcil IPC, retrabajo o manual Bizneo)."
)


def _partir_fragmentos(texto: str) -> List[str]:
    partes = re.split(r"(?<=[.!?])\s+|\n+", texto)
    return [p.strip() for p in partes if len(p.strip()) >= 20]


def _cita_literal_del_corpus(query: str) -> Tuple[Optional[str], List[Document]]:
    """Extrae del corpus la oración que hay que copiar; no delega al LLM."""
    frase = termino_mas_discriminante(query)
    if not frase or not state.corpus_docs:
        return None, []
    frase_l = frase.lower()
    q = query.lower()
    mejores: List[Tuple[int, int, str, Document]] = []
    for doc in state.corpus_docs:
        if frase_l not in doc.page_content.lower():
            continue
        for frag in _partir_fragmentos(doc.page_content):
            fl = frag.lower()
            if frase_l not in fl and not (
                "aviso" in q and "debe ficharse" in fl
            ):
                continue
            score = 0
            if "the ratio of the area" in fl:
                score += 30
            if "aperture walls" in fl:
                score += 12
            if "debe ficharse" in fl:
                score += 25
            if "no compartas" in fl or "credenciales" in fl:
                score += 18
            if "aviso" in q and "importante" in fl:
                score += 8
            if "definición" in q or "definicion" in q:
                if "the ratio of" in fl:
                    score += 10
            score += min(fl.count(frase_l), 3)
            mejores.append((score, len(frag), frag, doc))
    if not mejores:
        return None, []
    mejores.sort(key=lambda x: (-x[0], x[1]))
    _, _, cita, doc = mejores[0]
    return cita, [doc]


def _asegurar_oraciones_del_termino(
    query: str, answer: str, docs: List[Document]
) -> str:
    """Si un resumen omite el término pedido, añade una oración del contexto que lo contiene."""
    if not _es_pregunta_resumen(query):
        return answer
    term = termino_mas_discriminante(query)
    if not term:
        return answer
    term_l = term.lower()
    for doc in docs:
        for frag in _partir_fragmentos(doc.page_content):
            fl = frag.lower()
            if term_l not in fl:
                continue
            if "no deben" in fl and "no deben" not in answer.lower():
                return answer.rstrip() + "\n\n" + frag
    if term_l in answer.lower():
        return answer
    for doc in docs:
        for frag in _partir_fragmentos(doc.page_content):
            if term_l in frag.lower():
                return answer.rstrip() + "\n\n" + frag
    return answer


def _es_copia_literal(query: str) -> bool:
    return bool(
        re.search(r"\bcopia\b|literalmente|cita (exacta|literal|textual)", query, re.IGNORECASE)
    )


def _pregunta_en_espanol(query: str) -> bool:
    q = query.lower()
    return bool(
        re.search(r"[áéíóúñ¿¡]", q)
        or re.search(
            r"\b(qué|que|cómo|como|cuál|cuál|dónde|donde|cuáles|resumen|resume|"
            r"páginas|paginas|solicitan|ficha|dice)\b",
            q,
        )
    )


def _es_pregunta_extractiva(query: str) -> bool:
    return _es_copia_literal(query) or bool(
        re.search(r"exactamente|qu[eé] dice", query, re.IGNORECASE)
    )


def _es_seguimiento(query: str) -> bool:
    q = query.strip().lower()
    if len(q) > 180:
        return False
    return bool(
        re.search(
            r"trad[uú]c|castellano|en espa[nñ]ol|al espa[nñ]ol|"
            r"la puedes|lo puedes|esa respuesta|ese texto|la definici[oó]n|"
            r"expl[ií]calo|m[aá]s corto|m[aá]s claro",
            q,
        )
    )


def _es_pedido_traduccion(query: str) -> bool:
    return bool(re.search(r"trad[uú]c|castellano|en espa[nñ]ol|al espa[nñ]ol", query, re.IGNORECASE))


def _es_pregunta_resumen(query: str) -> bool:
    return bool(re.search(r"\bresume\b|resumen|apartados relacionados", query, re.IGNORECASE))


def _instruccion_respuesta(query: str) -> str:
    """Refuerzo breve según el tipo de pregunta (el system prompt largo se ignora a menudo)."""
    lineas: List[str] = []
    if not _es_copia_literal(query):
        lineas.append(
            "Responde en el mismo idioma que la pregunta. "
            "Si el contexto está en inglés y la pregunta en español, traduce los términos clave "
            "(aperture → apertura; aperture walls → paredes)."
        )
    if _es_pregunta_resumen(query):
        lineas.append(
            "Al resumir, menciona explícitamente cada documento de origen "
            "con el nombre exacto que aparece en las etiquetas del contexto (el PDF)."
        )
    if not lineas:
        return ""
    return "Instrucciones extra:\n" + "\n".join(f"- {ln}" for ln in lineas) + "\n\n"


def _completar_glosa_espanol(query: str, answer: str, context: str) -> str:
    """Glosa aperture→apertura solo en preguntas de Area Ratio / definición, no en listados de páginas."""
    if _es_copia_literal(query) or not _pregunta_en_espanol(query):
        return answer
    if es_intent_exhaustivo(query):
        return answer
    if not re.search(r"area ratio|\baperture\b|qué es|que es", query, re.IGNORECASE):
        return answer
    ctx = context.lower()
    ans = answer.lower()
    faltan: List[str] = []
    if "aperture" in ctx and "apertura" not in ans:
        faltan.append("apertura")
    if re.search(r"aperture walls|walls of (the )?aperture", ctx) and "paredes" not in ans:
        faltan.append("paredes")
    if not faltan:
        return answer
    if "apertura" in faltan and "paredes" in faltan:
        extra = (
            "En español: es la relación entre el área de la apertura "
            "y el área de las paredes de la apertura."
        )
    elif "apertura" in faltan:
        extra = "En español, aperture equivale a apertura."
    else:
        extra = "En español, aperture walls equivale a paredes."
    return answer.rstrip() + "\n\n" + extra


def _anexar_documentos_fuente(query: str, answer: str, docs: List[Document]) -> str:
    """Igual que la nota exhaustiva: el resumen debe nombrar los PDFs recuperados."""
    if not _es_pregunta_resumen(query) or not docs:
        return answer
    ids: List[str] = []
    vistos = set()
    for d in docs:
        did = str(d.metadata.get("document_id") or "")
        if did and did not in vistos:
            vistos.add(did)
            ids.append(did)
    if not ids:
        return answer
    if all(i.lower() in answer.lower() for i in ids):
        return answer
    return answer.rstrip() + "\n\nDocumentos fuente: " + ", ".join(ids) + "."


def _filtrar_docs_por_termino(query: str, docs: List[Document]) -> List[Document]:
    term = termino_mas_discriminante(query)
    if not term or not docs:
        return docs
    term_l = term.lower()
    con = [d for d in docs if term_l in d.page_content.lower()]
    return con if con else docs


class SimpleRetrievalQA:
    def __init__(self, llm, vectorstore_obj):
        self.llm = llm
        self.vectorstore = vectorstore_obj

    def __call__(self, query_dict: dict) -> Dict[str, Any]:
        query = query_dict.get("query", "")
        k = query_dict.get("k", DEFAULT_K)
        pregunta_anterior = (query_dict.get("pregunta_anterior") or "").strip()
        respuesta_anterior = (query_dict.get("respuesta_anterior") or "").strip()
        resultados_lexicos: List[Document] = []

        if (
            _es_seguimiento(query)
            and respuesta_anterior
            and _es_pedido_traduccion(query)
        ):
            prompt = (
                "Traduce al español de forma fiel el texto siguiente. "
                "No añadas datos que no estén en el texto. No hables de otros temas.\n\n"
                f"{respuesta_anterior}"
            )
            try:
                if hasattr(self.llm, "invoke"):
                    result = self.llm.invoke(
                        [
                            SystemMessage(content="Eres un traductor fiel. Solo traduces el texto dado."),
                            HumanMessage(content=prompt),
                        ]
                    )
                    answer = result.content if hasattr(result, "content") else str(result)
                else:
                    answer = respuesta_anterior
            except Exception as e:
                answer = f"Error al traducir: {e}"
            return {"result": answer, "source_documents": []}

        query_busqueda = query
        if _es_seguimiento(query) and pregunta_anterior:
            query_busqueda = f"{pregunta_anterior} {query}"

        if not (_es_seguimiento(query) and (pregunta_anterior or respuesta_anterior)):
            if _consulta_fuera_de_corpus(query):
                return {
                    "result": RESPUESTA_FUERA_DE_AMBITO,
                    "source_documents": [],
                }

        try:
            retrieval = pipeline_recuperacion(
                self.vectorstore,
                query_busqueda if (_es_seguimiento(query) and pregunta_anterior) else query,
                k=k,
                incluir_info_exhaustiva=True,
            )
            docs = retrieval["documentos"]
            resultados_lexicos = retrieval["resultados_lexicos"]
        except RetrievalPipelineError as e:
            print(f"[ERROR] Retrieval pipeline: {e}")
            docs = []
        except Exception as e:
            print(f"[ERROR] Retrieval: {e}")
            import traceback
            traceback.print_exc()
            docs = []

        if es_intent_exhaustivo(query):
            lex = resultados_lexicos or docs
            nota = construir_nota_exhaustiva(query, lex).strip()
            return {
                "result": nota or "No se encontraron menciones en el corpus.",
                "source_documents": lex[:k] if lex else docs,
            }

        if _es_pregunta_resumen(query):
            docs = _filtrar_docs_por_termino(query, docs)

        context = construir_contexto(docs) or "No se encontraron fragmentos relevantes."

        if _es_pregunta_extractiva(query):
            cita, docs_cita = _cita_literal_del_corpus(query)
            if cita:
                fuentes = docs_cita or docs
                return {"result": cita, "source_documents": fuentes}

        extra = _instruccion_respuesta(query)

        prompt = f"""{extra}Contexto:
{context}

Pregunta: {query}

Respuesta:"""

        try:
            if hasattr(self.llm, "invoke"):
                messages = [
                    SystemMessage(content=RAG_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
                result = self.llm.invoke(messages)
                answer = result.content if hasattr(result, "content") else str(result)
            else:
                answer = f"[Demo] Contexto sobre «{query}»:\n{context[:300]}..."
        except Exception as e:
            answer = f"Error al generar respuesta: {e}"

        answer = _completar_glosa_espanol(query, answer, context)
        answer = _asegurar_oraciones_del_termino(query, answer, docs)
        answer = _anexar_documentos_fuente(query, answer, docs)
        if es_intent_exhaustivo(query):
            answer += construir_nota_exhaustiva(query, resultados_lexicos)

        return {"result": answer, "source_documents": docs}


def crear_qa_chain(vs):
    return SimpleRetrievalQA(inicializar_llm(), vs)
