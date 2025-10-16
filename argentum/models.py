"""Core data models used throughout Argentum."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    """Agent role types."""

    MAKER = "maker"
    CHECKER = "checker"
    ADVISOR = "advisor"
    JUDGE = "judge"
    MODERATOR = "moderator"
    PARTICIPANT = "participant"


class MessageType(str, Enum):
    """Message types in agent conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class OrchestrationPattern(str, Enum):
    """Orchestration pattern types."""

    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    GROUP_CHAT = "group_chat"


class TerminationReason(str, Enum):
    """Reasons for conversation termination."""

    MAX_TURNS_REACHED = "max_turns_reached"
    CONSENSUS_REACHED = "consensus_reached"
    JUDGE_DECISION = "judge_decision"
    HUMAN_INTERVENTION = "human_intervention"
    ERROR = "error"
    TIMEOUT = "timeout"


class Message(BaseModel):
    """A message in an agent conversation."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: datetime.isoformat,
            UUID: str,
        },
    )

    id: UUID = Field(default_factory=uuid4)
    type: MessageType = MessageType.ASSISTANT
    sender: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Response from an agent."""

    agent_name: str
    content: str
    confidence: float | None = None
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrchestrationResult(BaseModel):
    """Result from an orchestration execution."""

    pattern: OrchestrationPattern
    messages: list[Message]
    final_outputs: list[AgentResponse]
    consensus: str | None = None
    termination_reason: TerminationReason
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float | None = None


class Task(BaseModel):
    """A task for agents to work on."""

    id: UUID = Field(default_factory=uuid4)
    description: str
    context: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
