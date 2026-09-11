"""Constantes y prompts del pipeline RAG (leídos de .env)."""
from __future__ import annotations

import os
import re

from .paths import ENV_PATH, UPLOAD_DIR as _UPLOAD_DIR, VECTORSTORE_PATH as _VECTORSTORE_PATH, ensure_data_dirs

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except ImportError:
    pass

ensure_data_dirs()

VECTORSTORE_PATH = str(_VECTORSTORE_PATH)
UPLOAD_DIR = str(_UPLOAD_DIR)
EXTENSIONES_PERMITIDAS = (".txt", ".pdf", ".md", ".docx")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

DEFAULT_K = int(os.getenv("DEFAULT_K", "8"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
# Traductor: google (rápido, online) | deepl (clave) | marian (local) | ollama
TRANSLATE_BACKEND = (os.getenv("TRANSLATE_BACKEND") or "google").strip().lower()
TRANSLATE_MODEL = (
    os.getenv("TRANSLATE_MODEL") or "Helsinki-NLP/opus-mt-en-es"
).strip()
TRANSLATE_NUM_PREDICT_MAX = int(os.getenv("TRANSLATE_NUM_PREDICT_MAX", "160"))
DEEPL_API_KEY = (os.getenv("DEEPL_API_KEY") or "").strip()
DEEPL_API_URL = (
    os.getenv("DEEPL_API_URL") or "https://api-free.deepl.com"
).rstrip("/")
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
STOPWORDS_BUSQUEDA = frozenset({
    "que", "qué", "cual", "cuál", "como", "cómo", "donde", "dónde",
    "cuando", "cuándo", "para", "por", "con", "sin", "sobre", "the", "and",
    "are", "was", "were", "has", "have", "this", "that", "from", "with",
    "los", "las", "del", "una", "uno", "unos", "unas", "por", "sus",
    "hay", "son", "ser", "esta", "está", "este", "esto", "esa", "eso",
    "muy", "mas", "más", "también", "tambien", "debe", "deben", "puede",
    "pueden", "entre", "hasta", "desde", "cada", "todo", "toda", "todos",
    "todas", "pero", "porque", "aunque", "hacia", "según", "segun",
    "not", "but", "for", "you", "your", "its", "than", "any", "all",
    "puedes", "puede", "quiero", "dame", "dime", "haz", "hacer",
    "era", "eran", "fue", "fui", "soy", "eres", "sido",
})
PALABRAS_INTENT_EXHAUSTIVO = frozenset({
    "aparece", "aparecen", "cuantas", "cuántas", "donde", "dónde", "lista",
    "menciones", "ocurrencias", "pagina", "página", "paginas", "páginas",
    "termino", "término", "todas", "veces",
})
TERMINOS_INTENT_QUERY = frozenset({
    "recomendaciones", "recomendacion", "recomendación",
    "resume", "resumen", "apartados", "apartado", "relacionados", "relacionado",
    "copia", "literalmente", "literal", "aviso", "importante",
    "dice", "exactamente", "documento", "definicion", "definición",
    "solicitan", "aparece", "aparecen",
})
