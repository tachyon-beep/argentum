"""Agent base classes and abstractions."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from argentum.models import AgentResponse, Message, Role


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    name: str
    role: Role = Role.PARTICIPANT
    persona: str
    model: str = "gpt-4"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    system_prompt: str | None = None
    tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    speaking_style: str | None = Field(
        default=None,
        description="High-level tone descriptor (e.g., 'boardroom', 'podcast', 'casual').",
    )
    speech_tags: list[str] = Field(
        default_factory=list,
        description="Additional tone modifiers (e.g., ['measured','warm']).",
    )
    tts_voice: str | None = Field(
        default=None,
        description="Suggested TTS voice identifier for this agent.",
    )


class Agent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, config: AgentConfig):
        """Initialize the agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.name = config.name
        self.role = config.role
        self.persona = config.persona
        self._message_history: list[Message] = []

    @abstractmethod
    async def generate_response(
        self,
        messages: list[Message],
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Generate a response based on conversation history and context.

        Args:
            messages: Conversation history
            context: Additional context for the response

        Returns:
            Agent's response
        """

    def receive_message(self, message: Message) -> None:
        """Receive and store a message.

        Args:
            message: Message to receive
        """
        self._message_history.append(message)

    def get_message_history(self) -> list[Message]:
        """Get the agent's message history.

        Returns:
            List of messages
        """
        return self._message_history.copy()

    def clear_history(self) -> None:
        """Clear the agent's message history."""
        self._message_history.clear()

    def get_capabilities(self) -> list[str]:
        """Get the agent's capabilities.

        Returns:
            List of capability descriptions
        """
        return self.config.tools.copy()

    def get_system_prompt(self) -> str:
        """Get the full system prompt for this agent.

        Returns:
            System prompt string
        """
        base_prompt = f"You are {self.name}, a {self.role.value}.\n\n"
        base_prompt += f"Persona: {self.persona}\n\n"

        if self.config.system_prompt:
            base_prompt += f"{self.config.system_prompt}\n\n"

        if self.config.speaking_style:
            base_prompt += (
                "Speaking Style: "
                f"{self.config.speaking_style}.\n\n"
            )

        if self.config.speech_tags:
            base_prompt += "Style notes: " + ", ".join(self.config.speech_tags) + "\n\n"

        if self.config.tools:
            base_prompt += f"You have access to the following tools: {', '.join(self.config.tools)}\n\n"

        base_prompt += "Please provide thoughtful, well-reasoned responses that align with your role and persona."

        return base_prompt

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"<Agent name='{self.name}' role='{self.role.value}'>"
