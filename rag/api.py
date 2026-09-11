"""Capa HTTP FastAPI. El pipeline vive en rag.ingest / retrieval / qa / store."""
from __future__ import annotations

import os
import secrets
import shutil
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel

from . import config, state
from .config import DEFAULT_K, RAG_ADMIN_TOKEN, UPLOAD_DIR, VECTORSTORE_PATH
from .errors import IngestError
from .ingest import (
    _destino_upload,
    cargar_archivo,
    crear_chunks,
    listar_archivos_subidos,
    nombre_archivo_seguro,
)
from .llm import inicializar_embeddings
from .paths import HTML_PATH, PROJECT_ROOT
from .qa import crear_qa_chain
from .retrieval import (
    _vectorstore_count,
    busqueda_lexica_exhaustiva,
    construir_contexto,
    metadata_json,
    pipeline_recuperacion,
    snippet_fuente,
)
from .store import (
    cargar_vectorstore_global,
    generar_resumen,
    reindexar_documentos,
    sincronizar_vectorstore,
)

# Reexportaciones para tests/eval_rag.py
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP


def exigir_token_admin(token: Optional[str]) -> None:
    if not RAG_ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="DELETE /limpiar deshabilitado. Define RAG_ADMIN_TOKEN en .env",
        )
    recibido = token or ""
    if not secrets.compare_digest(recibido, RAG_ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Token de administración inválido")


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
    pregunta_anterior: Optional[str] = None
    respuesta_anterior: Optional[str] = None


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
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="interfaz_web.html no encontrado")
    return FileResponse(str(HTML_PATH), media_type="text/html")


@app.get("/estado", response_model=EstadoResponse, tags=["General"])
async def obtener_estado():
    existe = state.vectorstore is not None
    total = getattr(state.vectorstore.index, "ntotal", None) if existe else None
    return EstadoResponse(
        estado="activo" if existe else "sin_documentos",
        vectorstore_cargado=existe,
        total_documentos=total,
        mensaje="Listo" if existe else "Sube documentos para comenzar",
    )


@app.get("/documentos", response_model=ListaDocumentosResponse, tags=["Documentos"])
async def listar_documentos():
    total = getattr(state.vectorstore.index, "ntotal", None) if state.vectorstore else None
    return ListaDocumentosResponse(archivos=listar_archivos_subidos(), total_chunks=total)


@app.get("/config", tags=["General"])
async def obtener_config():
    """Configuracion centralizada para la UI."""
    return {"default_k": DEFAULT_K}


@app.post("/buscar", tags=["RAG"])
async def buscar_chunks(request: PreguntaRequest):
    """Diagnóstico: fragmentos recuperados sin llamar al LLM."""
    if state.vectorstore is None:
        raise HTTPException(status_code=503, detail="No hay índice. Ejecuta POST /reindexar")
    k = request.k or DEFAULT_K
    docs = pipeline_recuperacion(state.vectorstore, request.pregunta, k=k)
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
    if state.qa_chain is None:
        raise HTTPException(status_code=503, detail="Sistema no inicializado. Sube documentos primero.")
    try:
        resultado = state.qa_chain(
            {
                "query": request.pregunta,
                "k": request.k or DEFAULT_K,
                "pregunta_anterior": request.pregunta_anterior,
                "respuesta_anterior": request.respuesta_anterior,
            }
        )
        fuentes = [
            {
                "contenido": snippet_fuente(request.pregunta, d.page_content),
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
    try:
        nombre = nombre_archivo_seguro(archivo.filename)
        file_path = str(_destino_upload(nombre))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(archivo.file, buffer)

        docs = cargar_archivo(file_path, nombre)
        chunks = crear_chunks(docs, nombre)
        if not chunks:
            raise HTTPException(status_code=400, detail="No se extrajo texto indexable del archivo")

        embeddings = inicializar_embeddings()
        if state.vectorstore is None:
            state.vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            nuevo = FAISS.from_documents(chunks, embeddings)
            state.vectorstore.merge_from(nuevo)

        state.vectorstore.save_local(VECTORSTORE_PATH)
        sincronizar_vectorstore(state.vectorstore)

        return DocumentoResponse(
            mensaje="Documento indexado correctamente",
            archivo=nombre,
            chunks_creados=len(chunks),
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except IngestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar documento: {e}")


@app.post("/reindexar", tags=["Admin"])
async def reindexar():
    try:
        resultado = reindexar_documentos()
        return {"mensaje": "Índice reconstruido", "timestamp": datetime.now().isoformat(), **resultado}
    except HTTPException:
        raise
    except IngestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
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
    exigir_token_admin(x_rag_admin_token)
    try:
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        state.vectorstore = None
        state.qa_chain = None
        state.corpus_docs.clear()
        state.bm25_index = None
        return {"mensaje": "Documentos e índice eliminados", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")




def __getattr__(name: str):
    """Compatibilidad con tests/eval_rag.py (api_rag.vectorstore, DEBUG_RAG, ...)."""
    aliases = {
        "vectorstore": lambda: state.vectorstore,
        "qa_chain": lambda: state.qa_chain,
        "_corpus_docs": lambda: state.corpus_docs,
        "_bm25_index": lambda: state.bm25_index,
        "DEBUG_RAG": lambda: config.DEBUG_RAG,
        "DIAG_PIPELINE": lambda: config.DIAG_PIPELINE,
    }
    if name in aliases:
        return aliases[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



if __name__ == "__main__":
    _root = PROJECT_ROOT
    _venv_py = _root / "venv" / "Scripts" / "python.exe"
    if _venv_py.exists() and Path(sys.executable).resolve() != _venv_py.resolve():
        os.execv(str(_venv_py), [str(_venv_py), str(_root / "launcher.py"), *sys.argv[1:]])
    os.chdir(_root)
    from launcher import main
    main()
