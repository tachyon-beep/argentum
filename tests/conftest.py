"""Test configuration and fixtures."""

from typing import Any

import pytest

from argentum.agents.base import AgentConfig
from argentum.llm.provider import LLMProvider
from argentum.models import AgentResponse, Role


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, response: str = "Mock response") -> None:
        self.response = response
        self.call_count = 0

    async def generate(self, messages: list[dict[str, str]], _temperature: float = 0.7, _max_tokens: int = 1000, **kwargs: Any) -> str:
        """Generate a mock response."""
        self.call_count += 1
        return self.response

    def get_model_name(self) -> str:
        """Get mock model name."""
        return "mock-model"

    def count_tokens(self, messages: list[Any]) -> int:
        """Count tokens in messages."""
        return sum(len(str(m)) for m in messages)

    async def generate_with_tools(self, messages: list[Any], tools: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Generate with tool support."""
        return AgentResponse(agent_name="mock", content=self.response)


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    """Provide a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def agent_config() -> AgentConfig:
    """Provide a default agent configuration."""
    return AgentConfig(
        name="Test Agent",
        role=Role.PARTICIPANT,
        persona="Test persona",
        model="gpt-4",
    )


@pytest.fixture
def multiple_agent_configs() -> list[AgentConfig]:
    """Provide multiple agent configurations."""
    return [
        AgentConfig(
            name="Agent 1",
            role=Role.PARTICIPANT,
            persona="First agent",
        ),
        AgentConfig(
            name="Agent 2",
            role=Role.PARTICIPANT,
            persona="Second agent",
        ),
        AgentConfig(
            name="Agent 3",
            role=Role.ADVISOR,
            persona="Third agent",
        ),
    ]
