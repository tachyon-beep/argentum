"""Session management for persisted conversations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from argentum.models import OrchestrationResult
from argentum.persistence.serializer import ConversationSerializer
from argentum.persistence.storage import ConversationStore


class ConversationSession:
    """Tracks orchestration results and persists them via a conversation store."""

    def __init__(
        self,
        store: ConversationStore,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.session_id = session_id or str(uuid4())
        self.metadata = metadata or {}
        self.history: list[OrchestrationResult] = []
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at
        self._loaded = False

    async def save(self, result: OrchestrationResult) -> None:
        """Persist an orchestration result to the backing store."""
        await self._ensure_loaded()
        self.history.append(result)
        self.updated_at = datetime.now(UTC)

        payload = {
            "session_id": self.session_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "results": [ConversationSerializer.serialize_result(item) for item in self.history],
        }

        await self.store.save_conversation(self.session_id, payload)

    async def load(self) -> None:
        """Load orchestration results from the backing store."""
        data = await self.store.load_conversation(self.session_id)

        self.metadata = data.get("metadata", {})
        self.created_at = ConversationSerializer._parse_datetime(data["created_at"]) if "created_at" in data else self.created_at
        self.updated_at = ConversationSerializer._parse_datetime(data["updated_at"]) if "updated_at" in data else self.updated_at
        self.history = [
            ConversationSerializer.deserialize_result(item) for item in data.get("results", [])
        ]
        self._loaded = True

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            await self.load()
        except FileNotFoundError:
            self._loaded = True

    def get_full_transcript(self) -> str:
        """Return a human-readable transcript for the entire session."""
        if not self.history:
            return "No conversations recorded."

        parts: list[str] = []
        for index, result in enumerate(self.history, start=1):
            parts.append(f"=== Conversation {index} ({result.pattern.value}) ===")
            for message in result.messages:
                if message.sender == "orchestrator":
                    continue
                parts.append(f"[{message.sender}]: {message.content}")
        return "\n".join(parts)
