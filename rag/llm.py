"""Embeddings, LLM y stubs de demostración."""
from __future__ import annotations

import os
import re
import sys

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import (
    DEEPL_API_KEY,
    DEEPL_API_URL,
    LLM_TEMPERATURE,
    TRANSLATE_BACKEND,
    TRANSLATE_MODEL,
    TRANSLATE_NUM_PREDICT_MAX,
)
from . import state

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


def ollama_esta_disponible(base_url: str) -> bool:
    try:
        import requests

        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False


def inicializar_llm():
    cfg = _llm_config()
    llm_type = cfg["type"]

    if llm_type == "ollama":
        if not OLLAMA_AVAILABLE:
            print("[AVISO] ChatOllama no está instalado — modo demo (fake)")
            return FakeChatModel()
        if not ollama_esta_disponible(cfg["ollama_url"]):
            print(
                f"[AVISO] Ollama no responde en {cfg['ollama_url']} — modo demo (fake). "
                "Reinstala Ollama y ejecuta `ollama pull llama3.2`, "
                "o define LLM_TYPE=openai / LLM_TYPE=fake en .env"
            )
            return FakeChatModel()
        print(f"[INFO] Ollama: {cfg['model']} ({cfg['ollama_url']})")
        return ChatOllama(
            base_url=cfg["ollama_url"],
            model=cfg["model"],
            temperature=LLM_TEMPERATURE,
            num_predict=350,
            repeat_penalty=1.45,
            keep_alive="30m",
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


def _num_predict_traduccion(texto: str) -> int:
    palabras = max(1, len((texto or "").split()))
    estimado = int(palabras * 1.6) + 24
    return max(48, min(TRANSLATE_NUM_PREDICT_MAX, estimado))


def _inicializar_llm_traduccion(texto: str = ""):
    """Fallback: mismo Ollama (o TRANSLATE_MODEL si es un tag Ollama), con pocos tokens."""
    cfg = _llm_config()
    if cfg["type"] != "ollama" or not OLLAMA_AVAILABLE:
        return inicializar_llm()
    if not ollama_esta_disponible(cfg["ollama_url"]):
        return FakeChatModel()
    modelo = TRANSLATE_MODEL if TRANSLATE_BACKEND == "ollama" else cfg["model"]
    if "/" in modelo:  # ruta HF — no es un tag de Ollama
        modelo = cfg["model"]
    return ChatOllama(
        base_url=cfg["ollama_url"],
        model=modelo,
        temperature=0,
        num_predict=_num_predict_traduccion(texto),
        num_ctx=2048,
        repeat_penalty=1.1,
        keep_alive="30m",
    )


def obtener_traductor_marian():
    """Carga lazy de MarianMT EN→ES (CPU). Independiente del dominio de los PDFs."""
    if state.traductor is not None:
        return state.traductor
    try:
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError(
            "transformers no disponible para el traductor MarianMT"
        ) from e
    modelo = TRANSLATE_MODEL or "Helsinki-NLP/opus-mt-en-es"
    print(f"[INFO] Cargando traductor MarianMT: {modelo}")
    state.traductor = pipeline(
        "translation",
        model=modelo,
        device=-1,
    )
    return state.traductor


def _partir_para_traducir(texto: str, max_chars: int = 400) -> list:
    """Parte textos largos (límites de Marian / APIs online)."""
    t = (texto or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    bloques: list = []
    for parrafo in t.split("\n"):
        parrafo = parrafo.strip()
        if not parrafo:
            continue
        if len(parrafo) <= max_chars:
            bloques.append(parrafo)
            continue
        actual = ""
        for frag in re.split(r"(?<=[.!?])\s+", parrafo):
            if not frag:
                continue
            if actual and len(actual) + 1 + len(frag) > max_chars:
                bloques.append(actual)
                actual = frag
            else:
                actual = f"{actual} {frag}".strip() if actual else frag
        if actual:
            bloques.append(actual)
    return bloques or [t[:max_chars]]


def _es_traduccion_invalida(texto: str) -> bool:
    t = (texto or "").strip().lower()
    if not t:
        return True
    if "error 500" in t or "server error" in t:
        return True
    if "that's an error" in t or "there was an error" in t:
        return True
    if t.startswith("<!doctype html") or t.startswith("<html"):
        return True
    return False


def _traducir_con_google(texto: str) -> str:
    from deep_translator import GoogleTranslator

    tr = GoogleTranslator(source="auto", target="es")
    partes = _partir_para_traducir(texto, max_chars=4500)
    salidas = []
    for p in partes:
        out = tr.translate(p)
        if _es_traduccion_invalida(out):
            raise RuntimeError(f"Google Translate devolvió error: {(out or '')[:80]}")
        salidas.append(out)
    return "\n".join(salidas).strip()


def _traducir_con_mymemory(texto: str) -> str:
    """API gratuita online (alternativa si Google limita)."""
    from deep_translator import MyMemoryTranslator

    tr = MyMemoryTranslator(source="en-GB", target="es-ES")
    partes = _partir_para_traducir(texto, max_chars=450)
    salidas = []
    for p in partes:
        out = tr.translate(p)
        if _es_traduccion_invalida(out):
            raise RuntimeError(f"MyMemory devolvió error: {(out or '')[:80]}")
        salidas.append(out)
    return "\n".join(salidas).strip()


def _traducir_con_deepl(texto: str) -> str:
    if not DEEPL_API_KEY:
        raise RuntimeError("Define DEEPL_API_KEY en .env para TRANSLATE_BACKEND=deepl")
    from deep_translator import DeeplTranslator

    tr = DeeplTranslator(
        api_key=DEEPL_API_KEY,
        source="en",
        target="es",
        use_free_api="api-free" in DEEPL_API_URL,
    )
    partes = _partir_para_traducir(texto, max_chars=4500)
    salidas = []
    for p in partes:
        out = tr.translate(p)
        if _es_traduccion_invalida(out):
            raise RuntimeError(f"DeepL devolvió error: {(out or '')[:80]}")
        salidas.append(out)
    return "\n".join(salidas).strip()


def _traducir_con_marian(texto: str) -> str:
    pipe = obtener_traductor_marian()
    partes = _partir_para_traducir(texto)
    salidas = []
    for parte in partes:
        out = pipe(parte, max_length=512, truncation=True)
        if isinstance(out, list) and out:
            salidas.append(
                out[0].get("translation_text") or out[0].get("generated_text") or ""
            )
        else:
            salidas.append(str(out))
    return "\n".join(s for s in salidas if s).strip() or texto


def _traducir_con_ollama(texto: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _inicializar_llm_traduccion(texto)
    if not hasattr(llm, "invoke"):
        return texto
    result = llm.invoke(
        [
            SystemMessage(content="Traductor EN→ES. Solo la traducción."),
            HumanMessage(
                content=(
                    "Traduce al español. Responde ÚNICAMENTE con la traducción, "
                    f"sin notas ni prefijos.\n\n{texto.strip()}"
                )
            ),
        ]
    )
    return (result.content if hasattr(result, "content") else str(result)).strip()


def traducir_en_es(texto: str) -> str:
    """Traduce a español según TRANSLATE_BACKEND, con cascada de respaldo."""
    t = (texto or "").strip()
    if not t:
        return t

    backend = TRANSLATE_BACKEND
    if backend == "deepl" and not DEEPL_API_KEY:
        print("[AVISO] DEEPL_API_KEY vacío; usando Google Translate")
        backend = "google"

    if backend == "google":
        intentos = [
            _traducir_con_google,
            _traducir_con_mymemory,
            _traducir_con_marian,
            _traducir_con_ollama,
        ]
    elif backend == "deepl":
        intentos = [
            _traducir_con_deepl,
            _traducir_con_google,
            _traducir_con_mymemory,
            _traducir_con_marian,
        ]
    elif backend == "marian":
        intentos = [_traducir_con_marian, _traducir_con_google, _traducir_con_ollama]
    elif backend == "ollama":
        intentos = [_traducir_con_ollama, _traducir_con_google]
    else:
        print(f"[AVISO] TRANSLATE_BACKEND={backend!r} desconocido; usando google")
        intentos = [
            _traducir_con_google,
            _traducir_con_mymemory,
            _traducir_con_marian,
            _traducir_con_ollama,
        ]

    ultimo_error = None
    for fn in intentos:
        try:
            out = fn(t)
            if out and out.strip() and not _es_traduccion_invalida(out):
                return out.strip()
            raise RuntimeError("respuesta vacía o inválida")
        except Exception as e:
            ultimo_error = e
            print(f"[AVISO] Traductor {fn.__name__} falló: {e}")
    if ultimo_error:
        print(f"[ERROR] Ningún traductor disponible: {ultimo_error}")
    return t
