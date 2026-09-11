"""Estado mutable del proceso: índice FAISS, corpus BM25 y cadena QA."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document

vectorstore = None
qa_chain = None
reranker = None
traductor = None
corpus_docs: List[Document] = []
bm25_index = None
