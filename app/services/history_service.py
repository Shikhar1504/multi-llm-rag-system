from __future__ import annotations

import logging
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.models.schemas import ChatMessage, SourceCitation


class HistoryService:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute("PRAGMA synchronous=NORMAL;")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)")

    def add_message(self, session_id: str, role: str, content: str, sources: list[SourceCitation] | None = None) -> None:
        payload = json.dumps([source.model_dump() for source in sources or []])
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, payload, datetime.now(timezone.utc).isoformat()),
            )

    def get_history(self, session_id: str) -> list[ChatMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content, sources, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

        messages: list[ChatMessage] = []
        for row in rows:
            try:
                raw_sources = json.loads(row["sources"] or "[]")
            except (TypeError, json.JSONDecodeError):
                raw_sources = []
            messages.append(
                ChatMessage(
                    role=row["role"],
                    content=row["content"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    sources=[SourceCitation(**source) for source in raw_sources],
                )
            )
        return messages
