from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings


@lru_cache(maxsize=16)
def _get_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def load_pdf_documents(file_path: Path, settings: Settings, source_name: str) -> list[Document]:
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    splitter = _get_splitter(settings.chunk_size, settings.chunk_overlap)
    chunks = splitter.split_documents(pages)
    normalized_chunks: list[Document] = []

    for index, chunk in enumerate(chunks):
        page_number = chunk.metadata.get("page")
        if page_number is None:
            page_number = chunk.metadata.get("page_number")

        page_content = chunk.page_content.strip()
        if not page_content:
            continue

        metadata = dict(chunk.metadata)
        metadata["source"] = source_name
        metadata["page"] = page_number
        metadata["chunk_index"] = index
        metadata["document_type"] = "pdf"
        metadata.pop("page_number", None)

        normalized_chunks.append(Document(page_content=page_content, metadata=metadata))

    return normalized_chunks
