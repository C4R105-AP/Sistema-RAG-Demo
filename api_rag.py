"""
API REST — RAG universal de producción (agnóstico multi-dominio).
Ingesta → chunking → embeddings → FAISS → retrieval → rerank → contexto → LLM.
Sin lógica especializada ni heurísticas por tipo de documento.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
from contextlib import asynccontextmanager
import contextlib
import os
import re
import secrets
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import numpy as np

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS_SRC = "huggingface"
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        HUGGINGFACE_EMBEDDINGS_SRC = "community"
    except ImportError:
        HuggingFaceEmbeddings = None
        HUGGINGFACE_EMBEDDINGS_SRC = None

try:
    from langchain_community.chat_models import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_community.llms import HuggingFacePipeline
    HUGGINGFACE_LLM_AVAILABLE = True
except ImportError:
    HUGGINGFACE_LLM_AVAILABLE = False

HUGGINGFACE_AVAILABLE = HUGGINGFACE_EMBEDDINGS_SRC is not None

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

VECTORSTORE_PATH = "vectorstore_faiss"
UPLOAD_DIR = "uploaded_docs"
EXTENSIONES_PERMITIDAS = (".txt", ".pdf", ".md", ".docx")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

DEFAULT_K = int(os.getenv("DEFAULT_K", "8"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
RETRIEVAL_MULTIPLIER = int(os.getenv("RETRIEVAL_MULTIPLIER", "4"))
CORPUS_SMALL_THRESHOLD = int(os.getenv("CORPUS_SMALL_THRESHOLD", "200"))
RRF_K = int(os.getenv("RRF_K", "60"))
MAX_CHARS_CONTEXTO = int(os.getenv("MAX_CHARS_CONTEXTO", "7000"))
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANK_WEIGHT_SIMILARITY = float(os.getenv("RERANK_WEIGHT_SIMILARITY", "0.7"))
RERANK_WEIGHT_CROSS = float(os.getenv("RERANK_WEIGHT_CROSS", "0.3"))

DEBUG_RAG = os.getenv("DEBUG_RAG", "false").lower() in ("1", "true", "yes")
DIAG_PIPELINE = os.getenv("DIAG_PIPELINE", "false").lower() in ("1", "true", "yes") or DEBUG_RAG
DEBUG_SIMILARITY_TOP_N = int(os.getenv("DEBUG_SIMILARITY_TOP_N", "30"))
DEBUG_PREVIEW_CHARS = int(os.getenv("DEBUG_PREVIEW_CHARS", "300"))
RERANK_MAX_CANDIDATES = int(os.getenv("RERANK_MAX_CANDIDATES", "64"))
MIN_CHUNKS_PER_DOC = int(os.getenv("MIN_CHUNKS_PER_DOC", "2"))
RAG_ADMIN_TOKEN = os.getenv("RAG_ADMIN_TOKEN", "").strip()

RAG_SYSTEM_PROMPT = (
    "Eres un asistente RAG (Retrieval-Augmented Generation) que responde exclusivamente usando el CONTEXTO proporcionado.\n\n"
    "Tu objetivo es responder de forma precisa, útil y fiel al contexto sin inventar información.\n\n"
    "---\n\n"
    "# REGLAS FUNDAMENTALES (OBLIGATORIAS)\n\n"
    "1. SOLO puedes usar información presente en el CONTEXTO.\n"
    "2. Nunca inventes, completes o asumas información externa.\n"
    "3. Si la información está en el contexto aunque sea parcial o fragmentada, debes usarla para construir la respuesta.\n"
    "4. Si el contexto es redundante, debes fusionarlo en una única respuesta coherente.\n"
    "5. Nunca repitas la misma idea con distintas palabras.\n\n"
    "---\n\n"
    "# REGLA CRÍTICA DE FALLBACK\n\n"
    "Está PROHIBIDO responder con:\n"
    "- 'no tengo información suficiente'\n"
    "- 'no se menciona en el contexto'\n"
    "- 'no hay información disponible'\n\n"
    "EXCEPTO si:\n"
    "- el CONTEXTO está completamente vacío\n\n"
    "Si hay cualquier información parcial relacionada, debes responderla.\n\n"
    "---\n\n"
    "# REGLA DE SÍNTESIS OBLIGATORIA\n\n"
    "Si el contexto contiene múltiples fragmentos similares:\n"
    "- combina la información en una sola explicación\n"
    "- elimina repeticiones\n"
    "- prioriza claridad sobre exhaustividad\n\n"
    "---\n\n"
    "# REGLA DE USO DEL CONTEXTO\n\n"
    "- Debes actuar como si el contexto fuera tu única fuente de verdad\n"
    "- Si la información está dispersa, debes reconstruirla\n"
    "- Si la pregunta pide resumir apartados, temas o secciones, nombra cada documento de origen "
    "usando exactamente el identificador de las etiquetas del contexto (nombre del PDF / document_id)\n\n"
    "---\n\n"
    "# REGLA DE IDIOMA\n\n"
    "- Responde en el mismo idioma que la pregunta\n"
    "- Si la pregunta está en español y el contexto en inglés, traduce la explicación y los términos clave "
    "(aperture → apertura, walls/paredes de la apertura → paredes, opening → apertura)\n"
    "- EXCEPCIÓN: si piden copia literal, cita textual o el texto exacto, reproduce el contexto sin traducir\n\n"
    "---\n\n"
    "# REGLA DE CONSISTENCIA\n\n"
    "- No contradigas el contexto\n"
    "- Si hay conflictos en el contexto, prioriza la información más repetida o más reciente (si está disponible)\n\n"
    "---\n\n"
    "# REGLA DE CONCEPTOS DISTINTOS\n\n"
    "Si la pregunta nombra un concepto concreto, responde solo a ese concepto.\n"
    "No uses umbrales, cifras o definiciones de un concepto distinto aunque aparezcan en el mismo contexto.\n\n"
    "---\n\n"
    "# REGLA DE ESTILO\n\n"
    "- Respuestas claras y directas\n"
    "- Sin redundancia\n"
    "- Sin repeticiones de ideas\n"
    "- Evita explicaciones largas innecesarias\n"
    "- Prioriza utilidad\n\n"
    "---\n\n"
    "# FORMATO DE RESPUESTA\n\n"
    "Responde siempre en formato:\n\n"
    "RESPUESTA:\n"
    "- explicación clara basada únicamente en el contexto\n"
)

PROMPT_RESUMEN = (
    "Resume el contenido proporcionado de forma clara y estructurada. "
    "Usa solo la información del contexto. "
    "Si falta información, no la inventes."
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

vectorstore = None
qa_chain = None
_reranker = None
_corpus_docs: List[Document] = []
_bm25_index = None

EXHAUSTIVE_PATTERNS = re.compile(
    r"(en qu[eé] p[aá]ginas?|d[oó]nde aparece|cu[aá]ntas veces|"
    r"todas las (veces|p[aá]ginas|menciones)|lista.*ocurrencias|"
    r"en qu[eé] parte|en qu[eé] secci[oó]n)",
    re.IGNORECASE,
)
LITERAL_PATTERNS = re.compile(
    r"(\bcopia\b|\bliteral|\bliteralmente\b|cita (exacta|literal|textual)|"
    r"\bdefine\b|definici[oó]n|qu[eé] es\b|qu[eé] significa\b)",
    re.IGNORECASE,
)
_STOPWORDS_BUSQUEDA = frozenset({
    "que", "qué", "cual", "cuál", "como", "cómo", "donde", "dónde",
    "cuando", "cuándo", "para", "por", "con", "sin", "sobre", "the", "and",
    "are", "was", "were", "has", "have", "this", "that", "from", "with",
    "los", "las", "del", "una", "uno", "unos", "unas", "por", "sus",
    "hay", "son", "ser", "esta", "está", "este", "esto", "esa", "eso",
    "muy", "mas", "más", "también", "tambien", "debe", "deben", "puede",
    "pueden", "entre", "hasta", "desde", "cada", "todo", "toda", "todos",
    "todas", "pero", "porque", "aunque", "hacia", "según", "segun",
    "not", "but", "for", "you", "your", "its", "than", "any", "all",
})
_PALABRAS_INTENT_EXHAUSTIVO = frozenset({
    "aparece", "aparecen", "cuantas", "cuántas", "donde", "dónde", "lista",
    "menciones", "ocurrencias", "pagina", "página", "paginas", "páginas",
    "termino", "término", "todas", "veces",
})
_TERMINOS_INTENT_QUERY = frozenset({
    "recomendaciones", "recomendacion", "recomendación",
    "resume", "resumen", "apartados", "apartado", "relacionados", "relacionado",
    "copia", "literalmente", "literal", "aviso", "importante",
    "dice", "exactamente", "documento", "definicion", "definición",
    "solicitan", "aparece", "aparecen",
})

# =============================================================================
# FAKE (modo demo sin LLM/embeddings reales)
# =============================================================================


class FakeChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        context = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context = msg.content[:500]
        text = (
            "[Modo demo — configura LLM_TYPE=ollama en .env]\n\n"
            f"Fragmentos recuperados:\n{context[:400]}..."
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _llm_type(self):
        return "fake"


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [np.random.rand(384).tolist() for _ in texts]

    def embed_query(self, text):
        return np.random.rand(384).tolist()


# =============================================================================
# INGESTA Y NORMALIZACIÓN DE TEXTO
# =============================================================================

_GARBAGE_PDF_MARKERS = (
    " ltd", " et0", "0bt", "bt\n", "et\n", "ltm0", " rg\n",
    "m905991", ".c2c", "ltd\n", "s0bt", "0.0.01k",
)


@contextlib.contextmanager
def silenciar_mupdf():
    try:
        import pymupdf
        if hasattr(pymupdf, "TOOLS"):
            pymupdf.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass
    stderr = sys.stderr
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = stderr


def limpiar_artefactos_pdf(texto: str) -> str:
    if not texto:
        return ""
    texto = texto.replace("\x00", " ")
    texto = re.sub(r"\b(BT|ET|rg|Tm|Td|Tj|TJ|re|f\*|n|W\*)\b", " ", texto)
    texto = re.sub(r"[^\S\n]+", " ", texto)
    return texto


def normalizar_texto(texto: str) -> str:
    """Normalización general: guiones de fin de línea, saltos, caracteres ilegibles."""
    if not texto:
        return ""
    texto = limpiar_artefactos_pdf(texto)
    texto = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\u024F]", " ", texto)
    texto = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", texto)
    texto = re.sub(r"(?<=\w)\n(?=\w)", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"[^\S\n]+", " ", texto)
    return texto.strip()


def es_texto_legible(texto: str) -> bool:
    """Filtra basura de streams gráficos PDF (artefactos, no contenido del documento)."""
    t = texto.strip()
    if len(t) < 20:
        return False
    letras = sum(c.isalpha() for c in t)
    if letras < 10:
        return False
    if letras / max(len(t), 1) < 0.15:
        return False
    if any(m in t.lower() for m in _GARBAGE_PDF_MARKERS):
        return False
    if len(re.findall(r"[a-zA-ZÁÉÍÓÚáéíóúÑñ]{3,}", t)) < 2:
        return False
    tokens = t.split()
    if len(tokens) > 5:
        numericos = sum(1 for tok in tokens if re.fullmatch(r"[\d.]+[a-z]?", tok) and len(tok) < 12)
        if numericos / len(tokens) > 0.65:
            return False
    return True


def cargar_pdf_limpio(file_path: str, filename: str) -> List[Document]:
    import pymupdf

    documentos: List[Document] = []
    with silenciar_mupdf():
        doc = pymupdf.open(file_path)
        try:
            total = len(doc)
            for page_num, page in enumerate(doc):
                partes: List[str] = []
                try:
                    for block in page.get_text("blocks", sort=True):
                        if len(block) < 5:
                            continue
                        if (block[6] if len(block) > 6 else 0) != 0:
                            continue
                        texto = normalizar_texto(str(block[4]))
                        if es_texto_legible(texto):
                            partes.append(texto)
                except Exception:
                    partes = []
                if not partes:
                    texto = normalizar_texto(page.get_text("text", sort=True))
                    if es_texto_legible(texto):
                        partes = [texto]
                if not partes:
                    continue
                documentos.append(
                    Document(
                        page_content="\n\n".join(partes),
                        metadata={"page": page_num + 1, "total_pages": total},
                    )
                )
        finally:
            doc.close()

    if not documentos:
        raise ValueError(f"No se extrajo texto legible de {filename}")
    return documentos


def cargar_archivo(file_path: str, filename: str) -> List[Document]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        try:
            return cargar_pdf_limpio(file_path, filename)
        except Exception as e:
            print(f"[ADVERTENCIA] Extracción PDF limpia falló ({filename}): {e}")
            with silenciar_mupdf():
                try:
                    docs = PyMuPDFLoader(file_path).load()
                except Exception:
                    docs = PyPDFLoader(file_path).load()
            for d in docs:
                d.page_content = normalizar_texto(d.page_content)
            return docs
    if ext in (".txt", ".md"):
        docs = TextLoader(file_path, encoding="utf-8").load()
        for d in docs:
            d.page_content = normalizar_texto(d.page_content)
        return docs
    if ext == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            docs = Docx2txtLoader(file_path).load()
            for d in docs:
                d.page_content = normalizar_texto(d.page_content)
            return docs
        except ImportError as e:
            raise HTTPException(
                status_code=400,
                detail="Para .docx instala: pip install python-docx docx2txt",
            ) from e
    raise HTTPException(
        status_code=400,
        detail=f"Formato no soportado. Usa: {', '.join(EXTENSIONES_PERMITIDAS)}",
    )


def preparar_paginas(docs: List[Document], document_id: str) -> List[Document]:
    preparados = []
    for doc in docs:
        texto = normalizar_texto(doc.page_content)
        if len(texto) < 20:
            continue
        preparados.append(
            Document(
                page_content=texto,
                metadata={
                    "document_id": document_id,
                    "page_number": doc.metadata.get("page"),
                },
            )
        )
    return preparados


def crear_chunks(docs: List[Document], document_id: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ": ", " "],
    )
    paginas = preparar_paginas(docs, document_id)
    if not paginas:
        return []
    raw = splitter.split_documents(paginas)
    chunks = []
    for i, chunk in enumerate(raw):
        chunk.metadata = {
            "document_id": document_id,
            "page_number": chunk.metadata.get("page_number"),
            "chunk_id": i,
        }
        chunks.append(chunk)
    return chunks


def listar_archivos_subidos() -> List[str]:
    if not os.path.exists(UPLOAD_DIR):
        return []
    return sorted(
        f for f in os.listdir(UPLOAD_DIR)
        if os.path.splitext(f)[1].lower() in EXTENSIONES_PERMITIDAS
    )


def nombre_archivo_seguro(filename: Optional[str]) -> str:
    if not filename or not str(filename).strip():
        raise HTTPException(status_code=400, detail="Nombre de archivo vacío")
    nombre = Path(str(filename).replace("\\", "/")).name
    if not nombre or nombre in {".", ".."} or ".." in nombre:
        raise HTTPException(status_code=400, detail="Nombre de archivo no válido")
    ext = os.path.splitext(nombre)[1].lower()
    if ext not in EXTENSIONES_PERMITIDAS:
        raise HTTPException(
            status_code=400,
            detail=f"Formatos: {', '.join(EXTENSIONES_PERMITIDAS)}",
        )
    return nombre


def _destino_upload(nombre: str) -> Path:
    raiz = Path(UPLOAD_DIR).resolve()
    destino = (raiz / nombre).resolve()
    if raiz != destino.parent:
        raise HTTPException(status_code=400, detail="Ruta de destino no permitida")
    return destino


def exigir_token_admin(token: Optional[str]) -> None:
    if not RAG_ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="DELETE /limpiar deshabilitado. Define RAG_ADMIN_TOKEN en .env",
        )
    recibido = token or ""
    if not secrets.compare_digest(recibido, RAG_ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Token de administración inválido")


# =============================================================================
# EMBEDDINGS Y LLM
# =============================================================================


def _llm_config() -> dict:
    return {
        "type": os.getenv("LLM_TYPE", "fake").lower(),
        "model": os.getenv("LLM_MODEL", "llama3.2"),
        "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
    }


def crear_embeddings_locales():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Instala sentence-transformers: .\\venv\\Scripts\\python.exe -m pip install sentence-transformers"
        ) from e
    if not HUGGINGFACE_AVAILABLE:
        raise RuntimeError("HuggingFaceEmbeddings no disponible")
    model = _llm_config()["embedding_model"]
    print(f"[INFO] Cargando embeddings: {model}")
    return HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"})


def inicializar_embeddings():
    cfg = _llm_config()
    if cfg["type"] == "openai" and OPENAI_AVAILABLE:
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return OpenAIEmbeddings(api_key=key)
    try:
        return crear_embeddings_locales()
    except Exception as e:
        print(f"[ADVERTENCIA] Embeddings locales no disponibles: {e}")
        print(f"[INFO] Python: {sys.executable}")
        print("[INFO] Usando embeddings fake — reindexa tras corregir el entorno")
        return FakeEmbeddings()


def inicializar_llm():
    cfg = _llm_config()
    llm_type = cfg["type"]

    if llm_type == "ollama":
        if not OLLAMA_AVAILABLE:
            print("[ERROR] ChatOllama no disponible")
            return FakeChatModel()
        print(f"[INFO] Ollama: {cfg['model']} ({cfg['ollama_url']})")
        return ChatOllama(
            base_url=cfg["ollama_url"],
            model=cfg["model"],
            temperature=LLM_TEMPERATURE,
            num_predict=500,
            repeat_penalty=1.3,
        )

    if llm_type == "openai":
        if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
            return FakeChatModel()
        return ChatOpenAI(model="gpt-4o-mini", temperature=LLM_TEMPERATURE, api_key=os.getenv("OPENAI_API_KEY"))

    if llm_type == "anthropic":
        if not ANTHROPIC_AVAILABLE or not os.getenv("ANTHROPIC_API_KEY"):
            return FakeChatModel()
        return ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    if llm_type == "huggingface" and HUGGINGFACE_LLM_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
            model = AutoModelForCausalLM.from_pretrained(cfg["model"], device_map="auto", torch_dtype="auto")
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=500,
                temperature=LLM_TEMPERATURE,
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"[ERROR] HuggingFace LLM: {e}")

    return FakeChatModel()


# =============================================================================
# RETRIEVAL, RERANK Y CONTEXTO
# =============================================================================


def obtener_reranker():
    """Carga lazy del cross-encoder (singleton)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "Instala sentence-transformers para el reranker: "
                "pip install sentence-transformers"
            ) from e
        print(f"[INFO] Cargando reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


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
    """Garantiza chunks léxicos top en el contexto (glosario/definiciones)."""
    lex_docs = busqueda_lexica_exhaustiva(query, vs)[: min(3, k)]
    ranked_docs = [doc for doc, _, _, _ in rankeados]
    merged: List[Document] = []
    vistos: set = set()
    for doc in lex_docs + ranked_docs:
        key = _chunk_key(doc)
        if key in vistos:
            continue
        vistos.add(key)
        merged.append(doc)
        if len(merged) >= k:
            break
    return merged


class RetrievalPipelineError(Exception):
    """Fallo crítico en retrieval (p. ej. índice vacío)."""
    pass


def _diag_log(mensaje: str) -> None:
    if DIAG_PIPELINE:
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
    return len(_corpus_docs)


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
    return [
        w for w in re.findall(r"\w+", query.lower())
        if len(w) > 2 and w not in _STOPWORDS_BUSQUEDA
    ]


def _frecuencia_documental(termino: str) -> int:
    t = termino.lower()
    return sum(1 for d in _corpus_docs if t in d.page_content.lower())


def termino_mas_discriminante(query: str) -> Optional[str]:
    """Término o bigrama de la query que aparece en menos chunks (más selectivo)."""
    terminos = [
        t for t in extraer_terminos_busqueda(query)
        if t not in _TERMINOS_INTENT_QUERY
    ]
    if not terminos or not _corpus_docs:
        return None
    n = len(_corpus_docs)
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
    docs = [d for d in _corpus_docs if term_l in d.page_content.lower()]
    docs.sort(key=lambda d: d.page_content.lower().count(term_l), reverse=True)
    return docs


def reconstruir_indice_bm25(vs) -> None:
    """Índice BM25 en memoria sobre el corpus completo."""
    global _corpus_docs, _bm25_index
    _corpus_docs = list(vs.docstore._dict.values())
    n = len(_corpus_docs)
    _diag_log(f"BM25 corpus len={n}")
    if n == 0:
        print("[ERROR] BM25: corpus vacío — reindexa con POST /reindexar")
        _bm25_index = None
        return
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[AVISO] rank_bm25 no instalado — pip install rank-bm25")
        _bm25_index = None
        return
    tokenizado = [doc.page_content.lower().split() for doc in _corpus_docs]
    vacios = sum(1 for t in tokenizado if not t)
    _diag_log(f"BM25 tokens: total={len(tokenizado)} listas_vacias={vacios}")
    if vacios == len(tokenizado):
        print("[ERROR] BM25: todos los chunks tienen tokens vacíos")
        _bm25_index = None
        return
    _bm25_index = BM25Okapi(tokenizado)
    print(f"[INFO] Índice BM25: {n} chunks (vacíos={vacios})")
    if DIAG_PIPELINE and _corpus_docs:
        _diag_log(_ejemplo_chunk_id(_corpus_docs[0], "BM25 corpus[0]"))


def _sincronizar_vectorstore(vs) -> None:
    global vectorstore, qa_chain
    vectorstore = vs
    reconstruir_indice_bm25(vs)
    qa_chain = crear_qa_chain(vs)


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
    return max(k * RETIEVAL_MULTIPLIER, k)


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
    if _bm25_index is None or not _corpus_docs:
        return []
    tokens = query.lower().split()
    if not tokens:
        return []
    try:
        puntajes = _bm25_index.get_scores(tokens)
        ordenados = sorted(enumerate(puntajes), key=lambda x: x[1], reverse=True)
        resultado: List[Document] = []
        for idx, score in ordenados:
            if score <= 0:
                break
            doc = _corpus_docs[idx]
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
    """Búsqueda léxica sobre todos los chunks del vectorstore (sin depender de _corpus_docs)."""
    terminos = extraer_terminos_busqueda(query)
    if not terminos:
        return []

    total_vs = _vectorstore_count(vs)
    if DEBUG_RAG:
        print(
            f"[DEBUG] Búsqueda léxica exhaustiva: len(_corpus_docs)={len(_corpus_docs)} | "
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
    if not terminos or not _corpus_docs:
        return []
    puntuados: List[Tuple[Document, int]] = []
    for doc in _corpus_docs:
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
        for d in _corpus_docs:
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
            for doc in _corpus_docs:
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
    if not DEBUG_RAG:
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
    if not DEBUG_RAG or not debe_ampliar_topn(query):
        return
    marcadores = ("1.1.3", "area of aperture walls")
    objetivos = [
        d for d in _corpus_docs
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


def construir_nota_exhaustiva(term: str, resultados_lexicos: list) -> str:
    """
    Construye la nota de cobertura exhaustiva a partir de los resultados léxicos.
    Solo se llama cuando EXHAUSTIVE_PATTERNS hace match.
    """
    terminos = [
        t for t in extraer_terminos_busqueda(term)
        if t not in _PALABRAS_INTENT_EXHAUSTIVO
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

    paginas = sorted(paginas)

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
    if DEBUG_RAG:
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

    if not _corpus_docs or len(_corpus_docs) != total_chunks:
        _diag_log(
            f"Corpus BM25 desincronizado (memoria={len(_corpus_docs)}, "
            f"índice={total_chunks}) — reconstruyendo"
        )
        reconstruir_indice_bm25(vs)

    top_n = _calcular_top_n(query, k, total_chunks)
    modo = _modo_retrieval_label(query, total_chunks)
    if DEBUG_RAG:
        exhaustivo = "SÍ" if es_intent_exhaustivo(query) else "NO"
        print(f'[DEBUG] Intent exhaustivo: {exhaustivo} — query: "{query}"')
        print(f"[DEBUG] debe_ampliar_topn() = {debe_ampliar_topn(query)}")
    _diag_log(f"Query={query!r} | modo={modo} | top_n={top_n} | k={k}")

    candidatos, sem_scored, bm25_docs, resultados_lexicos = _obtener_candidatos_rrf(query, vs, top_n)
    _diag_log(f"Candidatos pre-rerank (post-diversificación): {len(candidatos)}")

    if DEBUG_RAG:
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

    if DEBUG_RAG:
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
    """Si el contexto trae términos EN y la respuesta en ES los omite, añade la equivalencia."""
    if _es_copia_literal(query) or not _pregunta_en_espanol(query):
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


class SimpleRetrievalQA:
    def __init__(self, llm, vectorstore_obj):
        self.llm = llm
        self.vectorstore = vectorstore_obj

    def __call__(self, query_dict: dict) -> Dict[str, Any]:
        query = query_dict.get("query", "")
        k = query_dict.get("k", DEFAULT_K)
        resultados_lexicos: List[Document] = []

        try:
            retrieval = pipeline_recuperacion(
                self.vectorstore,
                query,
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

        context = construir_contexto(docs) or "No se encontraron fragmentos relevantes."
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
        answer = _anexar_documentos_fuente(query, answer, docs)
        if es_intent_exhaustivo(query):
            answer += construir_nota_exhaustiva(query, resultados_lexicos)

        return {"result": answer, "source_documents": docs}


def crear_qa_chain(vs):
    return SimpleRetrievalQA(inicializar_llm(), vs)


def cargar_vectorstore_global() -> bool:
    global vectorstore, qa_chain
    try:
        if os.path.exists(VECTORSTORE_PATH):
            embeddings = inicializar_embeddings()
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
            )
            _sincronizar_vectorstore(vectorstore)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Cargando vectorstore: {e}")
        return False


def reindexar_documentos() -> dict:
    global vectorstore, qa_chain

    archivos = listar_archivos_subidos()
    if not archivos:
        raise HTTPException(status_code=404, detail="No hay archivos en uploaded_docs/")

    embeddings = inicializar_embeddings()
    todos_chunks: List[Document] = []
    detalle = []

    for nombre in archivos:
        ruta = os.path.join(UPLOAD_DIR, nombre)
        docs = cargar_archivo(ruta, nombre)
        chunks = crear_chunks(docs, nombre)
        todos_chunks.extend(chunks)
        detalle.append({"archivo": nombre, "chunks": len(chunks)})

    if not todos_chunks:
        raise HTTPException(status_code=400, detail="No se extrajo texto de los archivos")

    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)

    vectorstore = FAISS.from_documents(todos_chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    _sincronizar_vectorstore(vectorstore)

    return {
        "archivos_procesados": len(detalle),
        "total_chunks": len(todos_chunks),
        "detalle": detalle,
    }


def generar_resumen(k: int = 12) -> dict:
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="No hay documentos indexados.")

    docs = recuperar_documentos(
        vectorstore, "resumen contenido principal temas conclusiones", k=k
    )
    if not docs:
        raise HTTPException(status_code=404, detail="No hay contenido para resumir.")

    contexto = construir_contexto(docs)
    llm = inicializar_llm()
    messages = [
        SystemMessage(content=PROMPT_RESUMEN),
        HumanMessage(content=f"Contexto:\n\n{contexto}\n\nGenera el resumen:"),
    ]

    if hasattr(llm, "invoke"):
        result = llm.invoke(messages)
        texto = result.content if hasattr(result, "content") else str(result)
    else:
        texto = f"[Demo] Fragmentos analizados: {len(docs)}"

    return {"resumen": texto, "fragmentos_usados": len(docs)}


# =============================================================================
# FASTAPI
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INICIO] RAG universal — cargando índice...")
    if cargar_vectorstore_global():
        print("[OK] Vectorstore cargado")
    else:
        print("[AVISO] Sin índice — sube documentos o POST /reindexar")
    yield
    print("[SHUTDOWN] Servidor detenido")


app = FastAPI(
    title="RAG Universal API",
    description="Buscador semántico + lector de contexto (multi-dominio)",
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PreguntaRequest(BaseModel):
    pregunta: str
    k: Optional[int] = DEFAULT_K


class RespuestaResponse(BaseModel):
    respuesta: str
    documentos_fuente: List[dict]
    timestamp: str


class EstadoResponse(BaseModel):
    estado: str
    vectorstore_cargado: bool
    total_documentos: Optional[int]
    mensaje: str


class DocumentoResponse(BaseModel):
    mensaje: str
    archivo: str
    chunks_creados: int
    timestamp: str


class ResumenRequest(BaseModel):
    k: Optional[int] = 12
    estilo: Optional[str] = None  # ignorado; compatibilidad con la UI


class ResumenResponse(BaseModel):
    resumen: str
    fragmentos_usados: int
    estilo: str = "general"
    timestamp: str


class ListaDocumentosResponse(BaseModel):
    archivos: List[str]
    total_chunks: Optional[int]


@app.get("/", tags=["General"])
async def root():
    return {
        "mensaje": "RAG Universal API",
        "version": "3.1.0",
        "interfaz": "/app",
        "endpoints": {
            "estado": "/estado",
            "documentos": "/documentos",
            "preguntar": "/preguntar (POST)",
            "buscar": "/buscar (POST)",
            "resumir": "/resumir (POST)",
            "subir_documento": "/subir-documento (POST)",
            "reindexar": "/reindexar (POST)",
            "recargar": "/recargar (POST)",
        },
    }


@app.get("/app", tags=["General"])
async def servir_interfaz():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interfaz_web.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="interfaz_web.html no encontrado")
    return FileResponse(html_path, media_type="text/html")


@app.get("/estado", response_model=EstadoResponse, tags=["General"])
async def obtener_estado():
    existe = vectorstore is not None
    total = getattr(vectorstore.index, "ntotal", None) if existe else None
    return EstadoResponse(
        estado="activo" if existe else "sin_documentos",
        vectorstore_cargado=existe,
        total_documentos=total,
        mensaje="Listo" if existe else "Sube documentos para comenzar",
    )


@app.get("/documentos", response_model=ListaDocumentosResponse, tags=["Documentos"])
async def listar_documentos():
    total = getattr(vectorstore.index, "ntotal", None) if vectorstore else None
    return ListaDocumentosResponse(archivos=listar_archivos_subidos(), total_chunks=total)


@app.get("/config", tags=["General"])
async def obtener_config():
    """Configuracion centralizada para la UI."""
    return {"default_k": DEFAULT_K}


@app.post("/buscar", tags=["RAG"])
async def buscar_chunks(request: PreguntaRequest):
    """Diagnóstico: fragmentos recuperados sin llamar al LLM."""
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="No hay índice. Ejecuta POST /reindexar")
    k = request.k or DEFAULT_K
    docs = recuperar_documentos(vectorstore, request.pregunta, k=k)
    return {
        "pregunta": request.pregunta,
        "total": len(docs),
        "chunks": [
            {
                "indice": i,
                "preview": d.page_content[:1000],
                "metadata": metadata_json(d),
            }
            for i, d in enumerate(docs, 1)
        ],
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/preguntar", response_model=RespuestaResponse, tags=["RAG"])
async def hacer_pregunta(request: PreguntaRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="Sistema no inicializado. Sube documentos primero.")
    try:
        resultado = qa_chain({"query": request.pregunta, "k": request.k or DEFAULT_K})
        fuentes = [
            {
                "contenido": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
                "metadata": metadata_json(d),
            }
            for d in resultado["source_documents"]
        ]
        return RespuestaResponse(
            respuesta=resultado["result"],
            documentos_fuente=fuentes,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")


@app.post("/resumir", response_model=ResumenResponse, tags=["RAG"])
async def resumir_documentos(request: ResumenRequest):
    try:
        r = generar_resumen(k=request.k or 12)
        return ResumenResponse(
            resumen=r["resumen"],
            fragmentos_usados=r["fragmentos_usados"],
            estilo=request.estilo or "general",
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al resumir: {e}")


@app.post("/subir-documento", response_model=DocumentoResponse, tags=["Documentos"])
async def subir_documento(archivo: UploadFile = File(...)):
    global vectorstore, qa_chain

    nombre = nombre_archivo_seguro(archivo.filename)

    try:
        file_path = str(_destino_upload(nombre))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(archivo.file, buffer)

        docs = cargar_archivo(file_path, nombre)
        chunks = crear_chunks(docs, nombre)
        if not chunks:
            raise HTTPException(status_code=400, detail="No se extrajo texto indexable del archivo")

        embeddings = inicializar_embeddings()
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            nuevo = FAISS.from_documents(chunks, embeddings)
            vectorstore.merge_from(nuevo)

        vectorstore.save_local(VECTORSTORE_PATH)
        _sincronizar_vectorstore(vectorstore)

        return DocumentoResponse(
            mensaje="Documento indexado correctamente",
            archivo=nombre,
            chunks_creados=len(chunks),
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar documento: {e}")


@app.post("/reindexar", tags=["Admin"])
async def reindexar():
    try:
        resultado = reindexar_documentos()
        return {"mensaje": "Índice reconstruido", "timestamp": datetime.now().isoformat(), **resultado}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al reindexar: {e}")


@app.post("/recargar", tags=["Admin"])
async def recargar_vectorstore():
    if cargar_vectorstore_global():
        return {"mensaje": "Vectorstore recargado", "timestamp": datetime.now().isoformat()}
    raise HTTPException(status_code=404, detail="No se encontró vectorstore")


@app.delete("/limpiar", tags=["Admin"])
async def limpiar_documentos(
    x_rag_admin_token: Optional[str] = Header(default=None, alias="X-RAG-Admin-Token"),
):
    global vectorstore, qa_chain, _bm25_index
    exigir_token_admin(x_rag_admin_token)
    try:
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        vectorstore = None
        qa_chain = None
        _corpus_docs.clear()
        _bm25_index = None
        return {"mensaje": "Documentos e índice eliminados", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")


if __name__ == "__main__":
    import uvicorn

    _root = Path(__file__).resolve().parent
    _venv_py = _root / "venv" / "Scripts" / "python.exe"
    if _venv_py.exists() and Path(sys.executable).resolve() != _venv_py.resolve():
        os.execv(str(_venv_py), [str(_venv_py), str(_root / "launcher.py"), *sys.argv[1:]])
    os.chdir(_root)
    from launcher import main
    main()
