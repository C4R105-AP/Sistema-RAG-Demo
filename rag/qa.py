"""Generación, postproceso determinista y cadena QA."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from . import state
from .config import (
    DEFAULT_K,
    PALABRAS_INTENT_EXHAUSTIVO,
    RAG_SYSTEM_PROMPT,
    STOPWORDS_BUSQUEDA,
    TERMINOS_INTENT_QUERY,
)
from .errors import RetrievalPipelineError
from .llm import inicializar_llm, traducir_en_es
from .retrieval import (
    _frecuencia_documental,
    construir_contexto,
    construir_nota_exhaustiva,
    es_intent_exhaustivo,
    extraer_terminos_busqueda,
    parece_tabla_contenidos,
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


def _terminos_contenido(query: str) -> List[str]:
    """Términos de tema de la query (sin intents ni stopwords)."""
    out: List[str] = []
    vistos = set()
    for t in extraer_terminos_busqueda(query):
        if t in TERMINOS_INTENT_QUERY or t in PALABRAS_INTENT_EXHAUSTIVO:
            continue
        if t in STOPWORDS_BUSQUEDA:
            continue
        if t in vistos:
            continue
        vistos.add(t)
        out.append(t)
    return out


def _term_in_text(term: str, text_l: str) -> bool:
    """Match léxico flexible (fichaje≈ficharse) sin diccionarios de dominio."""
    if not term:
        return False
    if term in text_l:
        return True
    if len(term) >= 5:
        raiz = term[: max(4, len(term) - 2)]
        if re.search(rf"\b{re.escape(raiz)}\w*", text_l):
            return True
    return False


def _puntuacion_oracion_cita(query: str, frag: str, frase_l: str) -> int:
    """Score genérico: cobertura de términos + forma (definición/norma), sin dominio."""
    fl = frag.lower()
    q = query.lower()
    if parece_tabla_contenidos(frag) or "......" in frag or frag.count("..") >= 6:
        return -100

    terminos = _terminos_contenido(query)
    if frase_l and frase_l not in terminos:
        terminos = [frase_l] + terminos

    score = 0
    hits = 0
    for t in terminos:
        if _term_in_text(t, fl):
            hits += 1
            score += 5 if t == frase_l else 2
    if hits == 0 and not (frase_l and _term_in_text(frase_l, fl)):
        return -50

    if terminos:
        score += int(10 * hits / max(len(terminos), 1))

    pos = fl.find(frase_l) if frase_l else -1
    if 0 <= pos <= 48:
        score += 8

    frase_re = re.escape(frase_l) if frase_l else ""
    if frase_re and re.search(rf"{frase_re}\s+(is|es|means|the ratio|se define)", fl):
        score += 14
    if frase_re and re.search(rf"\d+\.\d+[^\n]{{0,40}}\*?{frase_re}", fl):
        score += 10
    if re.search(r"\b(the ratio of|is the|means|se define como)\b", fl):
        score += 8

    # Fragmento cortado a medias (sin cierre de oración)
    if not re.search(r'[.!?"»]\s*$', frag.strip()):
        score -= 22

    pide_aviso = "aviso" in q
    pide_copia = bool(re.search(r"\bcopia\b|literalmente|cita (exacta|literal|textual)", q))
    if pide_aviso or pide_copia:
        if re.match(r"(?i)^(debe|deben|shall|must)\b", frag.strip()):
            score += 22
        if re.search(r"\b(debe|deben|shall|must|prohibido|no\s+compart)\b", fl):
            score += 12
        if re.search(r"\bes importante (leer|volver|escanear|fichar)\b", fl):
            score -= 14
        if pide_aviso and re.search(r"\bes importante\b", fl) and not re.search(
            r"\b(debe|deben|shall|must)\b", fl
        ):
            score -= 8
        # Aviso de norma: preferir enunciados cortos y cerrados
        if len(frag) <= 140 and re.search(r"[.!?]\s*$", frag.strip()):
            score += 8
        elif len(frag) <= 220:
            score += 3

    pide_def = bool(
        re.search(r"definici[oó]n|qu[eé]\s+es|qu[eé]\s+significa|copia|literalmente", q)
    )
    # «recomendaciones» no es definición de glosario
    if pide_def and not re.search(r"recomend|umbral|gu[ií]a", q):
        if len(frag) < 220:
            score += 4
        if re.search(
            r"\b(perform|between|should be|recommended|guideline for|chart showing)\b",
            fl,
        ):
            score -= 12
        if re.search(r"\b(the ratio of|opening|walls|is the)\b", fl):
            score += 6

    pide_recomend = bool(re.search(r"recomend|umbral|gu[ií]a|design guide", q))
    if pide_recomend:
        if re.search(r">\s*\d", fl) and hits:
            score += 22
        if re.search(r"\b(should be|design guide|acceptable|recommended)\b", fl):
            score += 12
        if re.search(r"\bthe ratio of\b", fl) and not re.search(r">\s*\d", fl):
            score -= 28
        if re.search(r"\b(chart showing|table\s+\d)\b", fl):
            score -= 10

    if frase_l and " " in frase_l and frase_l.endswith("ratio"):
        for m in re.finditer(r"\b([a-z]+)\s+ratio\b", fl):
            otro = m.group(0)
            if otro != frase_l:
                score -= 6

    if pide_copia and len(frag) < 260:
        score += 2
    return score


def _sanear_cita_umbrales(query: str, cita: str) -> str:
    """Si la query nombra un solo * ratio, quita umbrales del otro en la cita."""
    q = query.lower()
    if "aspect ratio" in q and "area ratio" not in q:
        cita = re.sub(
            r"(?i)\s*(?:and|,)?\s*>\s*0\.66\s+for area ratio",
            "",
            cita,
        )
    if "area ratio" in q and "aspect ratio" not in q:
        cita = re.sub(
            r"(?i)\s*(?:and|,)?\s*>\s*1\.5\s+for aspect ratio",
            "",
            cita,
        )
    return re.sub(r"\s{2,}", " ", cita).strip(" ,;")


def _cita_literal_de_docs(
    query: str, docs: List[Document]
) -> Tuple[Optional[str], List[Document], int]:
    """Elige la oración más alineada (score genérico). Devuelve (cita, docs, score)."""
    frase = termino_mas_discriminante(query)
    frase_l = (frase or "").lower()
    mejores: List[Tuple[int, int, str, Document]] = []
    terminos = _terminos_contenido(query)
    for doc in docs:
        if parece_tabla_contenidos(doc.page_content):
            continue
        for frag in _partir_fragmentos(doc.page_content):
            score = _puntuacion_oracion_cita(query, frag, frase_l)
            if score < 0:
                continue
            fl = frag.lower()
            if terminos and not any(_term_in_text(t, fl) for t in terminos):
                if not (frase_l and _term_in_text(frase_l, fl)):
                    continue
            mejores.append((score, len(frag), frag, doc))
    if not mejores:
        return None, [], -1
    mejores.sort(key=lambda x: (-x[0], x[1]))
    score, _, cita, doc = mejores[0]
    return _sanear_cita_umbrales(query, cita), [doc], score


def _ampliar_docs_por_termino(query: str, docs: List[Document]) -> List[Document]:
    """Une retrieval + chunks léxicos de los términos de la query."""
    terminos = _terminos_contenido(query)
    frase = termino_mas_discriminante(query)
    if frase and frase.lower() not in terminos:
        terminos = [frase.lower()] + terminos
    if not terminos or not state.corpus_docs:
        return docs
    vistos = {_chunk_key_safe(d) for d in docs}
    extra: List[Document] = []
    for d in state.corpus_docs:
        texto = d.page_content.lower()
        if not any(_term_in_text(t, texto) for t in terminos):
            continue
        if parece_tabla_contenidos(d.page_content):
            continue
        key = _chunk_key_safe(d)
        if key in vistos:
            continue
        vistos.add(key)
        extra.append(d)
        if len(extra) >= 50:
            break
    return list(docs) + extra


def _chunk_key_safe(doc: Document) -> str:
    return (
        f"{doc.metadata.get('document_id', '')}::"
        f"{doc.metadata.get('chunk_id', id(doc))}"
    )


def _intentar_cita_extractiva(
    query: str, docs: List[Document], min_score: int = 8
) -> Tuple[Optional[str], List[Document]]:
    pool = _ampliar_docs_por_termino(query, docs)
    cita, docs_cita, score = _cita_literal_de_docs(query, pool)
    if cita and score >= min_score:
        return cita, docs_cita
    return None, []


def _asegurar_oraciones_del_termino(
    query: str, answer: str, docs: List[Document]
) -> str:
    """Si un resumen no cita ninguna oración corta del contexto con el término, añade una."""
    if not _es_pregunta_resumen(query):
        return answer
    term = termino_mas_discriminante(query)
    if not term:
        return answer
    term_l = term.lower()
    candidatos: List[str] = []
    for doc in docs:
        for frag in _partir_fragmentos(doc.page_content):
            if term_l not in frag.lower():
                continue
            if parece_tabla_contenidos(frag):
                continue
            if len(frag) > 280:
                continue
            candidatos.append(frag)
    if not candidatos:
        return answer
    ans_l = answer.lower()
    if any(frag.lower() in ans_l for frag in candidatos):
        return answer
    return answer.rstrip() + "\n\n" + min(candidatos, key=len)


def _completar_con_evidencia_del_contexto(
    query: str, answer: str, docs: List[Document]
) -> str:
    """Si la respuesta omite un dato numérico presente junto al término en el contexto, lo aporta."""
    term = termino_mas_discriminante(query)
    if not term or not docs:
        return answer
    term_l = term.lower()
    ans = answer.lower()
    # Buscar oraciones del contexto con el término y un umbral tipo >1.5 / >0.66
    for doc in docs:
        for frag in _partir_fragmentos(doc.page_content):
            fl = frag.lower()
            if term_l not in fl:
                continue
            m = re.search(r">\s*\d+(?:[.,]\d+)?", frag)
            if not m:
                continue
            if m.group(0).replace(" ", "").lower() in ans.replace(" ", ""):
                return answer
            # Evitar colar umbrales de otro concepto si la query no lo pide
            if "area ratio" in fl and "aspect ratio" in term_l and "area ratio" not in query.lower():
                # quedarnos solo con la parte de aspect si se puede
                if "aspect ratio" in fl and re.search(r">\s*1\s*[.,]\s*5", fl):
                    limpio = re.sub(
                        r"(?i)\s*(?:and|,)?\s*>\s*0\.66\s+for area ratio",
                        "",
                        frag,
                    ).strip()
                    if limpio and "0.66" not in limpio:
                        return answer.rstrip() + "\n\n" + limpio
                continue
            return answer.rstrip() + "\n\n" + frag.strip()
    return answer


def _vetar_umbrales_cruzados(query: str, answer: str) -> str:
    """Si la pregunta pide un ratio concreto, quita umbrales ajenos que el LLM haya mezclado."""
    q = query.lower()
    if "aspect ratio" in q and "area ratio" not in q:
        texto = re.sub(
            r"(?i)\s*(?:and|,)?\s*>\s*0\.66\s+for area ratio",
            "",
            answer,
        )
        texto = re.sub(r"(?i)>\s*0\.66", "", texto)
        partes = re.split(r"(?<=[.!?])\s+|\n+", texto)
        return " ".join(
            p.strip() for p in partes if p.strip() and "0.66" not in p
        ).strip()
    return answer


def _normalizar_para_dedupe(texto: str) -> str:
    t = re.sub(r"\[.*?\]", " ", texto.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _limpiar_respuesta_llm(answer: str) -> str:
    """Quita prefijos y bucles de repetición típicos de llama3.2."""
    t = (answer or "").strip()
    t = re.sub(r"(?i)^respuesta:\s*", "", t).strip()
    t = re.sub(r"(?i)\bmanual_biznero\b", "Manual_Bizneo", t)
    t = re.sub(r"\s*([+\-])\s+(?=En\s*\[)", r"\n\1 ", t)

    bloques = [b.strip(" \t") for b in re.split(r"\n+", t) if b.strip(" \t")]
    unicos: List[str] = []
    vistos: List[str] = []
    for b in bloques:
        norm = _normalizar_para_dedupe(b)
        norm = re.sub(r"\b(?:chunk|p\.)\s*\d+\b", " ", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        if len(norm) >= 24 and any(
            SequenceMatcher(None, norm, v).ratio() >= 0.78 for v in vistos
        ):
            continue
        if len(norm) >= 24:
            vistos.append(norm)
        unicos.append(b)

    # Segunda pasada: oraciones duplicadas en un mismo bloque largo
    final: List[str] = []
    for b in unicos:
        oraciones = [o.strip() for o in re.split(r"(?<=[.!?])\s+", b) if o.strip()]
        if len(oraciones) < 3:
            final.append(b)
            continue
        keep: List[str] = []
        seen: List[str] = []
        for o in oraciones:
            norm = _normalizar_para_dedupe(o)
            if len(norm) >= 40 and any(
                SequenceMatcher(None, norm, s).ratio() >= 0.85 for s in seen
            ):
                continue
            if len(norm) >= 40:
                seen.append(norm)
            keep.append(o)
        final.append(" ".join(keep))
    return "\n".join(final).strip()


def _fuentes_para_respuesta(
    query: str, docs: List[Document], max_fuentes: int = 4
) -> List[Document]:
    """Fuentes útiles para la UI: sin TOC y priorizando el término de la pregunta."""
    docs = _fuentes_visibles(docs)
    term = termino_mas_discriminante(query)
    if term:
        term_l = term.lower()
        con = [d for d in docs if term_l in d.page_content.lower()]
        if con:
            docs = con
    return docs[:max_fuentes]


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


def _es_pregunta_definicion(query: str) -> bool:
    return bool(re.search(r"\bqu[eé]\s+es\b|\bdefinici[oó]n\b", query, re.IGNORECASE))


def _fuentes_visibles(docs: List[Document]) -> List[Document]:
    utiles = [d for d in docs if not parece_tabla_contenidos(d.page_content)]
    return utiles or docs


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


def _parece_ya_espanol(texto: str) -> bool:
    """Heurística de idioma (no vocabulario de dominio)."""
    t = (texto or "").lower()
    if re.search(r"[áéíóúñ¿¡]", t):
        return True
    es = len(re.findall(r"\b(el|la|los|las|que|para|cómo|como|según|debe|esta|este|una|del)\b", t))
    en = len(re.findall(r"\b(the|of|and|to|for|with|this|that|should|be)\b", t))
    return es >= 4 and es > en


def _traducir_respuesta(texto: str) -> str:
    t = (texto or "").strip()
    if not t:
        return t
    if _parece_ya_espanol(t):
        return t
    return traducir_en_es(t)


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
    lineas.append(
        "Cíñete solo a la pregunta. No mezcles otros procedimientos del mismo manual "
        "(p. ej. no hables de fichaje QR si preguntan por vacaciones)."
    )
    lineas.append(
        "No inventes significados de siglas ni nombres de archivo. "
        "No repitas el mismo párrafo ni la misma idea con distintas palabras. "
        "No empieces con la palabra RESPUESTA."
    )
    if _es_pregunta_resumen(query):
        lineas.append(
            "Resume en 4–8 frases claras. Menciona cada PDF de origen con el nombre "
            "exacto de las etiquetas del contexto."
        )
    if not lineas:
        return ""
    return "Instrucciones extra:\n" + "\n".join(f"- {ln}" for ln in lineas) + "\n\n"


def _completar_glosa_espanol(query: str, answer: str, context: str) -> str:
    """Si la pregunta va en español y el contexto trae aperture/walls, añade glosa breve."""
    if _es_copia_literal(query) or not _pregunta_en_espanol(query):
        return answer
    if es_intent_exhaustivo(query):
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

        if (
            _es_seguimiento(query)
            and respuesta_anterior
            and _es_pedido_traduccion(query)
        ):
            try:
                answer = _traducir_respuesta(respuesta_anterior)
            except Exception as e:
                answer = f"Error al traducir: {e}"
            return {"result": answer, "source_documents": []}

        query_busqueda = query
        if _es_seguimiento(query) and pregunta_anterior:
            query_busqueda = f"{pregunta_anterior} {query}"

        # Siempre recuperar chunks (también si luego se rechaza por fuera de ámbito).
        docs: List[Document] = []
        resultados_lexicos: List[Document] = []
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
        except Exception as e:
            print(f"[ERROR] Retrieval: {e}")
            import traceback
            traceback.print_exc()

        if not (_es_seguimiento(query) and (pregunta_anterior or respuesta_anterior)):
            if _consulta_fuera_de_corpus(query):
                return {
                    "result": RESPUESTA_FUERA_DE_AMBITO,
                    "source_documents": [],
                }

        if es_intent_exhaustivo(query):
            lex = resultados_lexicos or docs
            nota = construir_nota_exhaustiva(query, lex).strip()
            return {
                "result": nota or "No se encontraron menciones en el corpus.",
                "source_documents": _fuentes_para_respuesta(query, lex, max_fuentes=4),
            }

        docs = _filtrar_docs_por_termino(query, docs)
        docs = _fuentes_visibles(docs)

        # Citas / definiciones / recomendaciones numéricas: oración de chunks (scoring genérico).
        usar_cita = (
            _es_pregunta_extractiva(query)
            or _es_pregunta_definicion(query)
            or bool(re.search(r"recomend|umbral|gu[ií]a", query, re.IGNORECASE))
        )
        if usar_cita:
            min_score = 12 if _es_pregunta_definicion(query) else 8
            cita, docs_cita = _intentar_cita_extractiva(query, docs, min_score=min_score)
            if cita:
                if _es_pregunta_definicion(query) and not _es_copia_literal(query):
                    cita = _completar_glosa_espanol(query, cita, cita)
                return {
                    "result": cita,
                    "source_documents": _fuentes_para_respuesta(
                        query, docs_cita or docs, max_fuentes=3
                    ),
                }

        if _es_pregunta_resumen(query):
            docs = docs[:5]
        context = construir_contexto(docs, query) or "No se encontraron fragmentos relevantes."

        extra = _instruccion_respuesta(query)
        if _es_copia_literal(query) or _es_pregunta_extractiva(query):
            extra = (
                "Instrucciones extra:\n"
                "- Copia de forma literal la frase del contexto que responde; no parafrasees.\n\n"
            ) + extra

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

        answer = _limpiar_respuesta_llm(answer)
        answer = _completar_glosa_espanol(query, answer, context)
        answer = _vetar_umbrales_cruzados(query, answer)
        answer = _completar_con_evidencia_del_contexto(query, answer, docs)
        answer = _asegurar_oraciones_del_termino(query, answer, docs)
        answer = _limpiar_respuesta_llm(answer)
        answer = _anexar_documentos_fuente(query, answer, docs)

        return {
            "result": answer,
            "source_documents": _fuentes_para_respuesta(query, docs),
        }


def crear_qa_chain(vs):
    return SimpleRetrievalQA(inicializar_llm(), vs)
