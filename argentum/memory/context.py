"""Context and memory management."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from argentum.models import Message  # noqa: TCH001


class Context(BaseModel):
    """Shared context for agent conversations."""

    model_config = ConfigDict(
        json_encoders={
            datetime: datetime.isoformat,
            UUID: str,
        }
    )

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None

    def add_message(self, message: Message) -> None:
        """Add a message to the context.

        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.updated_at = datetime.now(UTC)

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get messages from the context.

        Args:
            limit: Maximum number of recent messages to return

        Returns:
            List of messages
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:]

    def get_messages_by_sender(self, sender: str) -> list[Message]:
        """Get messages from a specific sender.

        Args:
            sender: Sender name

        Returns:
            List of messages from the sender
        """
        return [msg for msg in self.messages if msg.sender == sender]

    def clear(self) -> None:
        """Clear all messages from the context."""
        self.messages.clear()
        self.summary = None
        self.updated_at = datetime.now(UTC)

    def get_conversation_length(self) -> int:
        """Get the total number of messages.

        Returns:
            Number of messages
        """
        return len(self.messages)

    def get_participants(self) -> set[str]:
        """Get unique participants in the conversation.

        Returns:
            Set of participant names
        """
        return {msg.sender for msg in self.messages}

    async def summarize(self, max_length: int = 500) -> str:
        """Generate a summary of the conversation.

        Args:
            max_length: Maximum length of the summary

        Returns:
            Conversation summary
        """
        if not self.messages:
            return "No messages in conversation."

        # Simple summarization - count messages per participant
        participants = self.get_participants()
        summary_parts = [
            f"Conversation with {len(participants)} participants:",
            f"Total messages: {len(self.messages)}",
        ]

        for participant in sorted(participants):
            count = len(self.get_messages_by_sender(participant))
            summary_parts.append(f"- {participant}: {count} messages")

        if self.messages:
            summary_parts.append(f"\nFirst message: {self.messages[0].content[:100]}...")
            summary_parts.append(f"Last message: {self.messages[-1].content[:100]}...")

        self.summary = "\n".join(summary_parts)
        return self.summary


class ConversationHistory(BaseModel):
    """Stores the history of a conversation."""

    model_config = ConfigDict(
        json_encoders={
            datetime: datetime.isoformat,
            UUID: str,
        }
    )

    context: Context
    agent_states: dict[str, Any] = Field(default_factory=dict)

    def add_agent_state(self, agent_name: str, state: Any) -> None:
        """Store state for a specific agent.

        Args:
            agent_name: Name of the agent
            state: State to store
        """
        self.agent_states[agent_name] = state

    def get_agent_state(self, agent_name: str) -> Any | None:
        """Get state for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent state or None
        """
        return self.agent_states.get(agent_name)
