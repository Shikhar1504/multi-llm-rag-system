from __future__ import annotations

from langchain_core.documents import Document

from app.rag.vector_stores.base import VectorStoreBackend


class PineconeBackend(VectorStoreBackend):
    def __init__(self, *, index_name: str, namespace: str, embedding_function, api_key: str | None) -> None:
        try:
            from langchain_pinecone import PineconeVectorStore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Pinecone backend requires langchain-pinecone to be installed.") from exc

        self._store = PineconeVectorStore(
            index_name=index_name,
            namespace=namespace,
            embedding=embedding_function,
            pinecone_api_key=api_key,
        )

    def add_documents(self, documents: list[Document]) -> None:
        if documents:
            self._store.add_documents(documents)

    def search(self, query: str, top_k: int, fetch_k: int | None = None) -> list[Document]:
        retriever = self._store.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)
