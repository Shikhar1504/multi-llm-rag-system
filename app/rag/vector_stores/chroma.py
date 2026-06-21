from __future__ import annotations

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from app.rag.vector_stores.base import VectorStoreBackend


class ChromaBackend(VectorStoreBackend):
    def __init__(self, *, persist_directory: str, collection_name: str, embedding_function) -> None:
        self._store = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        self._store.add_documents(documents)
        try:
            self._store.persist()
        except Exception:
            pass

    def search(self, query: str, top_k: int, fetch_k: int | None = None) -> list[Document]:
        retriever = self._store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": fetch_k or max(top_k * 2, top_k),
                "lambda_mult": 0.5,
            },
        )
        return retriever.invoke(query)
