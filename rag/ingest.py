"""Carga de archivos, limpieza de PDF y chunking."""
from __future__ import annotations

import contextlib
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EXTENSIONES_PERMITIDAS, UPLOAD_DIR
from .errors import IngestError

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
            raise IngestError(
                "Para .docx instala: pip install python-docx docx2txt",
            ) from e
    raise IngestError(
        f"Formato no soportado. Usa: {', '.join(EXTENSIONES_PERMITIDAS)}",
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
        raise IngestError("Nombre de archivo vacío")
    nombre = Path(str(filename).replace("\\", "/")).name
    if not nombre or nombre in {".", ".."} or ".." in nombre:
        raise IngestError("Nombre de archivo no válido")
    ext = os.path.splitext(nombre)[1].lower()
    if ext not in EXTENSIONES_PERMITIDAS:
        raise IngestError(
            f"Formatos: {', '.join(EXTENSIONES_PERMITIDAS)}",
        )
    return nombre


def _destino_upload(nombre: str) -> Path:
    raiz = Path(UPLOAD_DIR).resolve()
    destino = (raiz / nombre).resolve()
    if raiz != destino.parent:
        raise IngestError("Ruta de destino no permitida")
    return destino
