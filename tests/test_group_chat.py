"""Tests for group chat orchestration."""

import pytest
from conftest import MockLLMProvider

from argentum.agents.llm_agent import LLMAgent
from argentum.coordination.chat_manager import ChatManager
from argentum.models import OrchestrationPattern, TerminationReason
from argentum.orchestration.group_chat import GroupChatOrchestrator


class TestGroupChatOrchestrator:
    """Test group chat orchestration pattern."""

    @pytest.mark.asyncio
    async def test_group_chat_execution(self, multiple_agent_configs):
        """Test group chat execution."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        chat_manager = ChatManager(max_turns=6, min_turns=2)
        orchestrator = GroupChatOrchestrator(chat_manager=chat_manager)

        result = await orchestrator.execute(agents=agents, task="Test discussion")

        assert result.pattern == OrchestrationPattern.GROUP_CHAT
        assert len(result.final_outputs) == 6
        assert result.termination_reason == TerminationReason.MAX_TURNS_REACHED

    @pytest.mark.asyncio
    async def test_group_chat_with_default_manager(self, multiple_agent_configs):
        """Test group chat with default chat manager."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        orchestrator = GroupChatOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Test")

        assert result.pattern == OrchestrationPattern.GROUP_CHAT
        assert len(result.final_outputs) > 0

    @pytest.mark.asyncio
    async def test_group_chat_turn_taking(self, multiple_agent_configs):
        """Test turn-taking in group chat."""
        provider = MockLLMProvider(response="Turn response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        chat_manager = ChatManager(max_turns=6, min_turns=2)
        orchestrator = GroupChatOrchestrator(chat_manager=chat_manager)

        result = await orchestrator.execute(agents=agents, task="Test")

        # Check that each agent spoke at least once in 6 turns
        agent_names = [out.agent_name for out in result.final_outputs]
        assert "Agent 1" in agent_names
        assert "Agent 2" in agent_names
        assert "Agent 3" in agent_names

    @pytest.mark.asyncio
    async def test_group_chat_error_handling(self, multiple_agent_configs):
        """Test error handling during group chat."""

        class ErrorProvider(MockLLMProvider):
            def __init__(self):
                super().__init__()
                self.call_number = 0

            async def generate(self, messages, temperature=0.7, max_tokens=1000, **kwargs):
                self.call_number += 1
                if self.call_number == 2:
                    raise ValueError("Simulated error")
                return "Response"

        provider = ErrorProvider()
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        chat_manager = ChatManager(max_turns=4, min_turns=2)
        orchestrator = GroupChatOrchestrator(chat_manager=chat_manager)

        result = await orchestrator.execute(agents=agents, task="Test")

        # Should have some outputs despite error
        assert len(result.final_outputs) > 0
        # Should have error message in context
        error_messages = [msg for msg in result.messages if msg.metadata.get("error")]
        assert len(error_messages) > 0

    @pytest.mark.asyncio
    async def test_group_chat_consensus(self, multiple_agent_configs):
        """Test consensus generation."""
        provider = MockLLMProvider(response="Final point")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        chat_manager = ChatManager(max_turns=3)
        orchestrator = GroupChatOrchestrator(chat_manager=chat_manager)

        result = await orchestrator.execute(agents=agents, task="Test")

        assert result.consensus is not None
        assert len(result.consensus) > 0

    @pytest.mark.asyncio
    async def test_group_chat_statistics(self, multiple_agent_configs):
        """Test statistics in group chat result."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        chat_manager = ChatManager(max_turns=6)
        orchestrator = GroupChatOrchestrator(chat_manager=chat_manager)

        result = await orchestrator.execute(agents=agents, task="Test")

        assert "statistics" in result.metadata
        stats = result.metadata["statistics"]
        assert stats["total_turns"] == 6
        assert stats["unique_speakers"] == 3

    @pytest.mark.asyncio
    async def test_group_chat_duration(self, multiple_agent_configs):
        """Test duration tracking in group chat."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        orchestrator = GroupChatOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Test")

        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0
