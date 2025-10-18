"""Conversation storage backends."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ConversationStore(ABC):
    """Abstract base class for storing conversations."""

    @abstractmethod
    async def save_conversation(self, conversation_id: str, data: dict[str, Any]) -> None:
        """Persist a conversation payload."""

    @abstractmethod
    async def load_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Load a conversation payload."""

    @abstractmethod
    async def list_conversations(self, filters: dict[str, Any] | None = None) -> list[str]:
        """List conversation identifiers."""

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> None:
        """Remove a stored conversation."""


class JSONFileStore(ConversationStore):
    """File-based storage using JSON documents."""

    def __init__(self, base_path: Path | str = Path("./conversations")) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._extension = ".json"

    async def save_conversation(self, conversation_id: str, data: dict[str, Any]) -> None:
        file_path = self._resolve_path(conversation_id)
        await asyncio.to_thread(self._write_json, file_path, data)

    async def load_conversation(self, conversation_id: str) -> dict[str, Any]:
        file_path = self._resolve_path(conversation_id)
        if not file_path.exists():
            raise FileNotFoundError(f"Conversation '{conversation_id}' not found")
        return await asyncio.to_thread(self._read_json, file_path)

    async def list_conversations(self, filters: dict[str, Any] | None = None) -> list[str]:
        def _list_ids() -> list[str]:
            if not self.base_path.exists():
                return []
            ids: list[str] = []
            for path in self.base_path.glob(f"*{self._extension}"):
                ids.append(path.stem)
            ids.sort()
            return ids

        # Filters are currently ignored; included for future extensibility.
        _ = filters
        return await asyncio.to_thread(_list_ids)

    async def delete_conversation(self, conversation_id: str) -> None:
        file_path = self._resolve_path(conversation_id)
        if file_path.exists():
            await asyncio.to_thread(file_path.unlink)

    def _resolve_path(self, conversation_id: str) -> Path:
        return self.base_path / f"{conversation_id}{self._extension}"

    @staticmethod
    def _write_json(file_path: Path, data: dict[str, Any]) -> None:
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _read_json(file_path: Path) -> dict[str, Any]:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
