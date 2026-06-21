from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class VectorStoreBackend(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int, fetch_k: int | None = None) -> list[Document]:
        raise NotImplementedError
