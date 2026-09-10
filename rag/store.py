"""Carga, sincronización y reconstrucción del índice FAISS."""
from __future__ import annotations

import os
import shutil
from typing import List

from fastapi import HTTPException
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from . import state
from .config import PROMPT_RESUMEN, UPLOAD_DIR, VECTORSTORE_PATH
from .ingest import cargar_archivo, crear_chunks, listar_archivos_subidos
from .llm import inicializar_embeddings, inicializar_llm
from .retrieval import construir_contexto, reconstruir_indice_bm25, recuperar_documentos


def sincronizar_vectorstore(vs) -> None:
    state.vectorstore = vs
    reconstruir_indice_bm25(vs)
    from .qa import crear_qa_chain
    state.qa_chain = crear_qa_chain(vs)


def cargar_vectorstore_global() -> bool:
    try:
        if os.path.exists(VECTORSTORE_PATH):
            embeddings = inicializar_embeddings()
            vs = FAISS.load_local(
                VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
            )
            sincronizar_vectorstore(vs)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Cargando vectorstore: {e}")
        return False


def reindexar_documentos() -> dict:
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

    vs = FAISS.from_documents(todos_chunks, embeddings)
    vs.save_local(VECTORSTORE_PATH)
    sincronizar_vectorstore(vs)

    return {
        "archivos_procesados": len(detalle),
        "total_chunks": len(todos_chunks),
        "detalle": detalle,
    }


def generar_resumen(k: int = 12) -> dict:
    if state.vectorstore is None:
        raise HTTPException(status_code=503, detail="No hay documentos indexados.")

    docs = recuperar_documentos(
        state.vectorstore, "resumen contenido principal temas conclusiones", k=k
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
