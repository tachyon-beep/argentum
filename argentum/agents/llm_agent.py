"""LLM-powered agent implementation."""

from typing import Any

from argentum.agents.base import Agent, AgentConfig
from argentum.llm.provider import LLMProvider
from argentum.models import AgentResponse, Message, MessageType


class LLMAgent(Agent):
    """An agent powered by a Large Language Model."""

    def __init__(self, config: AgentConfig, provider: LLMProvider):
        """Initialize the LLM agent.

        Args:
            config: Agent configuration
            provider: LLM provider instance
        """
        super().__init__(config)
        self.provider = provider

    async def generate_response(
        self,
        messages: list[Message],
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Generate a response using the LLM provider.

        Args:
            messages: Conversation history
            context: Additional context for the response

        Returns:
            Agent's response
        """
        # Build the prompt with system message
        system_prompt = self.get_system_prompt()

        # Add context if provided
        if context:
            context_str = "\n\nAdditional Context:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
            system_prompt += context_str

        system_message = Message(
            type=MessageType.SYSTEM,
            sender="system",
            content=system_prompt,
        )

        # Prepare messages for the LLM
        all_messages = [system_message, *messages]

        # Convert to provider format
        provider_messages = [{"role": msg.type.value, "content": msg.content} for msg in all_messages]

        # Generate response
        response_content = await self.provider.generate(
            messages=provider_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Create and return agent response
        return AgentResponse(
            agent_name=self.name,
            content=response_content,
            metadata={
                "model": self.config.model,
                "temperature": self.config.temperature,
                "role": self.role.value,
            },
        )
