"""Persistent memory helpers for individual agents."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MAX_ENTRIES = 200


def _now() -> datetime:
    """Return current UTC time."""
    return datetime.now(UTC)


@dataclass(slots=True)
class MemoryEntry:
    """A single memory entry for an agent."""

    content: str
    timestamp: datetime = field(default_factory=_now)
    topic: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to a JSON-friendly dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Deserialize entry from stored dictionary."""
        timestamp_str = data.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else _now()
        return cls(
            content=data.get("content", ""),
            timestamp=timestamp,
            topic=data.get("topic"),
            metadata=data.get("metadata", {}),
        )


class AgentMemory:
    """In-memory representation of an agent's historical statements."""

    def __init__(
        self,
        agent_name: str,
        *,
        entries: list[MemoryEntry] | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.agent_name = agent_name
        self.max_entries = max_entries
        self._entries: list[MemoryEntry] = list(entries) if entries else []

    def add_entry(
        self,
        content: str,
        *,
        topic: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a new statement in memory."""
        entry = MemoryEntry(content=content, topic=topic, metadata=metadata or {})
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            # Keep the most recent entries
            self._entries = self._entries[-self.max_entries :]
        return entry

    def recent_entries(
        self,
        *,
        limit: int = 5,
        topic: str | None = None,
    ) -> list[MemoryEntry]:
        """Return the most recent statements, optionally filtered by topic."""
        relevant = self._entries if topic is None else [entry for entry in self._entries if entry.topic == topic]
        return list(relevant[-limit:])

    def build_prompt_fragment(
        self,
        *,
        limit: int = 5,
        topic: str | None = None,
    ) -> str | None:
        """Return a formatted string describing recent statements."""
        entries = self.recent_entries(limit=limit, topic=topic)
        if not entries:
            return None

        lines = [
            "You have previously stated:",
            *[f"- {entry.content}" for entry in entries],
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory for persistence."""
        return {
            "agent_name": self.agent_name,
            "max_entries": self.max_entries,
            "entries": [entry.to_dict() for entry in self._entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMemory:
        """Create memory instance from stored data."""
        entries = [MemoryEntry.from_dict(item) for item in data.get("entries", [])]
        return cls(
            agent_name=data.get("agent_name", "unknown"),
            entries=entries,
            max_entries=data.get("max_entries", DEFAULT_MAX_ENTRIES),
        )


class AgentMemoryStore:
    """Simple file-backed memory store for agents."""

    def __init__(
        self,
        base_path: str | Path = "agent_memories",
        *,
        max_entries_per_agent: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_entries_per_agent = max_entries_per_agent
        self._cache: dict[str, AgentMemory] = {}

    def get_memory(self, agent_name: str) -> AgentMemory:
        """Retrieve memory for an agent, loading from disk if necessary."""
        if agent_name in self._cache:
            return self._cache[agent_name]

        file_path = self._file_path(agent_name)
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            memory = AgentMemory.from_dict(raw)
        else:
            memory = AgentMemory(agent_name, max_entries=self.max_entries_per_agent)

        self._cache[agent_name] = memory
        return memory

    def record_statement(
        self,
        agent_name: str,
        content: str,
        *,
        topic: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Record a new statement and persist it immediately."""
        memory = self.get_memory(agent_name)
        entry = memory.add_entry(content, topic=topic, metadata=metadata)
        self._persist(memory)
        return entry

    def get_history_prompt(
        self,
        agent_name: str,
        *,
        limit: int = 5,
        topic: str | None = None,
    ) -> str | None:
        """Return prompt fragment describing agent's prior statements."""
        memory = self.get_memory(agent_name)
        return memory.build_prompt_fragment(limit=limit, topic=topic)

    def _persist(self, memory: AgentMemory) -> None:
        file_path = self._file_path(memory.agent_name)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(memory.to_dict(), f, indent=2)

    def _file_path(self, agent_name: str) -> Path:
        safe_name = agent_name.lower().replace(" ", "_")
        return self.base_path / f"{safe_name}.json"
