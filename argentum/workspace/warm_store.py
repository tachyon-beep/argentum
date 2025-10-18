"""Warm cache store for semantic/session highlights."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


class WarmCacheStore:
    """SQLite FTS-backed store for session highlights."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS entries
                USING fts5(
                    session_id,
                    item_type,
                    agent,
                    text,
                    metadata
                )
                """
            )

    def replace_session_entries(self, session_id: str, items: Iterable[dict[str, Any]]) -> None:
        """Replace indexed entries for a session."""
        rows = []
        for item in items:
            if not item.get("text"):
                continue
            rows.append(
                (
                    session_id,
                    item.get("type") or item.get("item_type") or "unknown",
                    item.get("agent"),
                    item.get("text"),
                    json.dumps(item.get("metadata") or {}),
                )
            )

        with self._connect() as conn:
            conn.execute("DELETE FROM entries WHERE session_id = ?", (session_id,))
            if rows:
                conn.executemany(
                    "INSERT INTO entries(session_id, item_type, agent, text, metadata) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Search the warm cache using FTS query syntax."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT session_id, item_type, agent, text, metadata FROM entries WHERE entries MATCH ? LIMIT ?",
                (query, limit),
            )
            results: list[dict[str, Any]] = []
            for row in cursor:
                metadata = {}
                if row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        metadata = {"raw": row["metadata"]}
                results.append(
                    {
                        "session_id": row["session_id"],
                        "type": row["item_type"],
                        "agent": row["agent"],
                        "text": row["text"],
                        "metadata": metadata,
                    }
                )
            return results

    def list_recent(self, *, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent entries without filtering."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT session_id, item_type, agent, text, metadata FROM entries ORDER BY rowid DESC LIMIT ?",
                (limit,),
            )
            results: list[dict[str, Any]] = []
            for row in cursor:
                metadata = {}
                if row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        metadata = {"raw": row["metadata"]}
                results.append(
                    {
                        "session_id": row["session_id"],
                        "type": row["item_type"],
                        "agent": row["agent"],
                        "text": row["text"],
                        "metadata": metadata,
                    }
                )
            return results

    def delete_sessions(self, session_ids: Iterable[str]) -> None:
        """Remove entries for the given session identifiers."""
        ids = list(session_ids)
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            conn.execute(f"DELETE FROM entries WHERE session_id IN ({placeholders})", ids)

    def clear(self) -> None:
        """Remove all indexed entries."""
        with self._connect() as conn:
            conn.execute("DELETE FROM entries")

    def count_entries(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM entries")
            row = cursor.fetchone()
            return int(row[0]) if row else 0
