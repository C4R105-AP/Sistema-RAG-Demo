"""Embeddings, LLM y stubs de demostración."""
from __future__ import annotations

import os
import sys

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import LLM_TEMPERATURE

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
