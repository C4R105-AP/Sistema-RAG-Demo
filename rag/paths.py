"""Rutas del proyecto. Independientes del directorio de trabajo."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

_DATA_OVERRIDE = os.getenv("RAG_DATA_DIR", "").strip()
DATA_DIR = Path(_DATA_OVERRIDE).resolve() if _DATA_OVERRIDE else (PROJECT_ROOT / "data")

UPLOAD_DIR = DATA_DIR / "uploaded_docs"
VECTORSTORE_PATH = DATA_DIR / "vectorstore_faiss"
WEB_DIR = PROJECT_ROOT / "web"
HTML_PATH = WEB_DIR / "interfaz_web.html"
ENV_PATH = PROJECT_ROOT / ".env"


def ensure_data_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
