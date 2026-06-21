from __future__ import annotations

import hashlib
import logging
import sqlite3
from sqlite3 import IntegrityError
from pathlib import Path

from fastapi import UploadFile
from langchain_core.documents import Document

from app.core.config import Settings
from app.rag.ingestion import load_pdf_documents
from app.rag.vector_stores.base import VectorStoreBackend


class DocumentService:
    def __init__(self, *, settings: Settings, vector_store: VectorStoreBackend) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._registry_db = settings.registry_db_path
        self._registry_db.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
        self._initialize_registry()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._registry_db)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_registry(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute("PRAGMA synchronous=NORMAL;")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _checksum(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _lookup_checksum(self, sha256: str) -> dict[str, str] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT document_id, file_name, chunk_count FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        if not row:
            return None
        return {"document_id": row["document_id"], "file_name": row["file_name"], "chunk_count": row["chunk_count"]}

    def _register_document(self, *, document_id: str, file_name: str, sha256: str, chunk_count: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO documents(document_id, file_name, sha256, chunk_count, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
                (document_id, file_name, sha256, chunk_count),
            )

    def _delete_document(self, sha256: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE sha256 = ?", (sha256,))

    async def ingest_upload(self, upload_file: UploadFile) -> dict[str, object]:
        suffix = Path(upload_file.filename or "document.pdf").suffix or ".pdf"
        temp_path = self._settings.uploads_dir / f"{hashlib.sha1((upload_file.filename or 'document').encode()).hexdigest()}{suffix}"
        self._logger.info("Starting ingestion for file %s", upload_file.filename or temp_path.name)

        try:
            with temp_path.open("wb") as handle:
                while chunk := await upload_file.read(1024 * 1024):
                    handle.write(chunk)

            file_hash = self._checksum(temp_path)
            existing = self._lookup_checksum(file_hash)
            if existing:
                self._logger.info("Duplicate ingestion detected for file %s", upload_file.filename or temp_path.name)
                return {
                    "document_id": existing["document_id"],
                    "file_name": existing["file_name"],
                    "status": "already_indexed",
                    "chunk_count": int(existing["chunk_count"]),
                    "skipped": True,
                }

            document_id = file_hash[:16]
            try:
                self._register_document(
                    document_id=document_id,
                    file_name=upload_file.filename or temp_path.name,
                    sha256=file_hash,
                    chunk_count=0,
                )
            except IntegrityError:
                existing = self._lookup_checksum(file_hash)
                if existing:
                    self._logger.info("Duplicate ingestion detected for file %s during insert", upload_file.filename or temp_path.name)
                    return {
                        "document_id": existing["document_id"],
                        "file_name": existing["file_name"],
                        "status": "already_indexed",
                        "chunk_count": int(existing["chunk_count"]),
                        "skipped": True,
                    }
                raise

            chunks = load_pdf_documents(temp_path, self._settings, upload_file.filename or temp_path.name)
            try:
                self._vector_store.add_documents(chunks)
            except Exception:
                self._delete_document(file_hash)
                raise

            with self._connect() as connection:
                connection.execute(
                    "UPDATE documents SET chunk_count = ? WHERE sha256 = ?",
                    (len(chunks), file_hash),
                )

            self._logger.info("Finished ingestion for file %s", upload_file.filename or temp_path.name)
            return {
                "document_id": document_id,
                "file_name": upload_file.filename or temp_path.name,
                "status": "indexed",
                "chunk_count": len(chunks),
                "skipped": False,
            }
        finally:
            temp_path.unlink(missing_ok=True)
