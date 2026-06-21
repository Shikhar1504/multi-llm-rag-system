from __future__ import annotations

from app.core.config import Settings
from app.rag.embeddings import build_embeddings
from app.rag.vector_stores.chroma import ChromaBackend
from app.rag.vector_stores.pinecone import PineconeBackend


def build_vector_store(settings: Settings):
    embeddings = build_embeddings(settings)

    if settings.vector_store_backend == "pinecone":
        return PineconeBackend(
            index_name=settings.pinecone_index_name,
            namespace=settings.pinecone_namespace,
            embedding_function=embeddings,
            api_key=settings.pinecone_api_key,
        )

    return ChromaBackend(
        persist_directory=str(settings.chroma_dir),
        collection_name="rag_documents",
        embedding_function=embeddings,
    )
