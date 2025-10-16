"""Tests for agent implementations."""

import pytest
from conftest import MockLLMProvider

from argentum.agents.base import AgentConfig
from argentum.agents.llm_agent import LLMAgent
from argentum.models import Message, MessageType, Role


class TestAgentConfig:
    """Test agent configuration."""

    def test_agent_config_creation(self):
        """Test creating an agent config."""
        config = AgentConfig(
            name="Test Agent",
            role=Role.ADVISOR,
            persona="Test persona",
        )
        assert config.name == "Test Agent"
        assert config.role == Role.ADVISOR
        assert config.persona == "Test persona"

    def test_agent_config_defaults(self):
        """Test default values in agent config."""
        config = AgentConfig(
            name="Test",
            persona="Test",
        )
        assert config.role == Role.PARTICIPANT
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_agent_config_with_tools(self):
        """Test agent config with tools."""
        config = AgentConfig(
            name="Tool Agent",
            persona="Has tools",
            tools=["web_search", "calculator"],
        )
        assert len(config.tools) == 2
        assert "web_search" in config.tools

    def test_agent_config_temperature_bounds(self):
        """Test temperature validation."""
        with pytest.raises(ValueError):
            AgentConfig(name="Test", persona="Test", temperature=3.0)

    def test_agent_config_max_tokens_positive(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError):
            AgentConfig(name="Test", persona="Test", max_tokens=-1)


class TestLLMAgent:
    """Test LLM agent implementation."""

    @pytest.mark.asyncio
    async def test_generate_response(self, agent_config):
        """Test generating a response."""
        provider = MockLLMProvider(response="Test response")
        agent = LLMAgent(config=agent_config, provider=provider)

        messages = [
            Message(
                type=MessageType.USER,
                sender="user",
                content="Test message",
            )
        ]

        response = await agent.generate_response(messages)

        assert response.agent_name == "Test Agent"
        assert response.content == "Test response"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_response_with_context(self, agent_config):
        """Test generating a response with additional context."""
        provider = MockLLMProvider(response="Contextual response")
        agent = LLMAgent(config=agent_config, provider=provider)

        messages = [Message(type=MessageType.USER, sender="user", content="Question")]
        context = {"background": "Important info", "constraints": ["Be brief"]}

        response = await agent.generate_response(messages, context=context)

        assert response.content == "Contextual response"
        assert response.agent_name == "Test Agent"

    def test_agent_system_prompt(self, agent_config):
        """Test system prompt generation."""
        provider = MockLLMProvider()
        agent = LLMAgent(config=agent_config, provider=provider)

        prompt = agent.get_system_prompt()

        assert "Test Agent" in prompt
        assert "Test persona" in prompt
        assert agent_config.role.value in prompt

    def test_agent_system_prompt_with_tools(self):
        """Test system prompt includes tools."""
        config = AgentConfig(name="Tool Agent", persona="Has tools", tools=["web_search", "calculator"])
        provider = MockLLMProvider()
        agent = LLMAgent(config=config, provider=provider)

        prompt = agent.get_system_prompt()

        assert "web_search" in prompt
        assert "calculator" in prompt

    def test_message_history(self, agent_config):
        """Test message history tracking."""
        provider = MockLLMProvider()
        agent = LLMAgent(config=agent_config, provider=provider)

        msg1 = Message(type=MessageType.USER, sender="user", content="Hello")
        msg2 = Message(type=MessageType.ASSISTANT, sender="agent", content="Hi")

        agent.receive_message(msg1)
        agent.receive_message(msg2)

        history = agent.get_message_history()
        assert len(history) == 2
        assert history[0].content == "Hello"
        assert history[1].content == "Hi"

    def test_clear_history(self, agent_config):
        """Test clearing message history."""
        provider = MockLLMProvider()
        agent = LLMAgent(config=agent_config, provider=provider)

        msg = Message(type=MessageType.USER, sender="user", content="Test")
        agent.receive_message(msg)

        agent.clear_history()
        assert len(agent.get_message_history()) == 0

    def test_get_capabilities(self):
        """Test getting agent capabilities."""
        config = AgentConfig(name="Test", persona="Test", tools=["tool1", "tool2"])
        provider = MockLLMProvider()
        agent = LLMAgent(config=config, provider=provider)

        capabilities = agent.get_capabilities()
        assert len(capabilities) == 2
        assert "tool1" in capabilities
        assert "tool2" in capabilities

    def test_agent_repr(self, agent_config):
        """Test agent string representation."""
        provider = MockLLMProvider()
        agent = LLMAgent(config=agent_config, provider=provider)

        repr_str = repr(agent)
        assert "Test Agent" in repr_str
        assert agent_config.role.value in repr_str
