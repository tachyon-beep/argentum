"""Serialization helpers for conversation persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from argentum.memory.context import Context
from argentum.models import (
    AgentResponse,
    Message,
    MessageType,
    OrchestrationPattern,
    OrchestrationResult,
    TerminationReason,
)


class ConversationSerializer:
    """Utility methods to convert conversations to and from serialisable payloads."""

    @staticmethod
    def serialize_result(result: OrchestrationResult) -> dict[str, Any]:
        """Convert an orchestration result into a JSON-friendly payload."""
        return {
            "id": str(uuid4()),
            "pattern": result.pattern.value,
            "consensus": result.consensus,
            "termination_reason": result.termination_reason.value,
            "duration_seconds": result.duration_seconds,
            "metadata": result.metadata,
            "messages": [ConversationSerializer.serialize_message(msg) for msg in result.messages],
            "final_outputs": [
                ConversationSerializer.serialize_agent_response(response) for response in result.final_outputs
            ],
        }

    @staticmethod
    def deserialize_result(payload: dict[str, Any]) -> OrchestrationResult:
        """Reconstruct an orchestration result from persisted data."""
        messages = [ConversationSerializer.deserialize_message(item) for item in payload.get("messages", [])]
        responses = [
            ConversationSerializer.deserialize_agent_response(item) for item in payload.get("final_outputs", [])
        ]

        return OrchestrationResult(
            pattern=OrchestrationPattern(payload["pattern"]),
            messages=messages,
            final_outputs=responses,
            consensus=payload.get("consensus"),
            termination_reason=TerminationReason(payload["termination_reason"]),
            metadata=payload.get("metadata", {}),
            duration_seconds=payload.get("duration_seconds"),
        )

    @staticmethod
    def serialize_context(context: Context) -> dict[str, Any]:
        """Convert a context model into a JSON-friendly payload."""
        return {
            "id": str(context.id),
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat(),
            "summary": context.summary,
            "metadata": context.metadata,
            "messages": [ConversationSerializer.serialize_message(msg) for msg in context.messages],
        }

    @staticmethod
    def deserialize_context(payload: dict[str, Any]) -> Context:
        """Reconstruct a context model from persisted data."""
        context = Context(
            id=UUID(payload["id"]),
            created_at=ConversationSerializer._parse_datetime(payload["created_at"]),
            updated_at=ConversationSerializer._parse_datetime(payload["updated_at"]),
            summary=payload.get("summary"),
            metadata=payload.get("metadata", {}),
            messages=[ConversationSerializer.deserialize_message(item) for item in payload.get("messages", [])],
        )
        return context

    @staticmethod
    def serialize_message(message: Message) -> dict[str, Any]:
        """Convert a message into a plain dictionary."""
        return {
            "id": str(message.id),
            "type": message.type.value,
            "sender": message.sender,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "metadata": message.metadata,
        }

    @staticmethod
    def deserialize_message(payload: dict[str, Any]) -> Message:
        """Reconstruct a message from a dictionary."""
        return Message(
            id=UUID(payload["id"]),
            type=MessageType(payload["type"]),
            sender=payload["sender"],
            content=payload["content"],
            timestamp=ConversationSerializer._parse_datetime(payload["timestamp"]),
            metadata=payload.get("metadata", {}),
        )

    @staticmethod
    def serialize_agent_response(response: AgentResponse) -> dict[str, Any]:
        """Convert an agent response into a plain dictionary."""
        return {
            "agent_name": response.agent_name,
            "content": response.content,
            "confidence": response.confidence,
            "citations": response.citations,
            "metadata": response.metadata,
        }

    @staticmethod
    def deserialize_agent_response(payload: dict[str, Any]) -> AgentResponse:
        """Reconstruct an agent response from a dictionary."""
        return AgentResponse(
            agent_name=payload["agent_name"],
            content=payload["content"],
            confidence=payload.get("confidence"),
            citations=payload.get("citations", []),
            metadata=payload.get("metadata", {}),
        )

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        """Parse an ISO formatted datetime string."""
        dt = datetime.fromisoformat(value)
        # Normalise to aware datetime in UTC to match default model configuration.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
