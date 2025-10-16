"""Tests for orchestration patterns."""

from typing import Any

import pytest
from conftest import MockLLMProvider

from argentum.agents.llm_agent import LLMAgent
from argentum.models import AgentResponse, Message, OrchestrationPattern, TerminationReason
from argentum.orchestration.concurrent import ConcurrentOrchestrator
from argentum.orchestration.sequential import SequentialOrchestrator


class TestSequentialOrchestrator:
    """Test sequential orchestration pattern."""

    @pytest.mark.asyncio
    async def test_sequential_execution(self, multiple_agent_configs):
        """Test sequential agent execution."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        orchestrator = SequentialOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Test task")

        assert result.pattern == OrchestrationPattern.SEQUENTIAL
        assert len(result.final_outputs) == 3
        assert result.termination_reason == TerminationReason.MAX_TURNS_REACHED

    @pytest.mark.asyncio
    async def test_sequential_with_string_task(self, multiple_agent_configs):
        """Test sequential execution with string task."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        orchestrator = SequentialOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Simple string task")

        assert result.pattern == OrchestrationPattern.SEQUENTIAL
        assert len(result.final_outputs) == 3

    @pytest.mark.asyncio
    async def test_sequential_preserves_order(self, multiple_agent_configs):
        """Test that agents execute in specified order."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        orchestrator = SequentialOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Test")

        # Check that outputs are in order
        assert result.final_outputs[0].agent_name == "Agent 1"
        assert result.final_outputs[1].agent_name == "Agent 2"
        assert result.final_outputs[2].agent_name == "Agent 3"

    @pytest.mark.asyncio
    async def test_sequential_consensus(self, multiple_agent_configs):
        """Test consensus generation in sequential pattern."""
        provider = MockLLMProvider(response="Final response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        orchestrator = SequentialOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Test")

        # Consensus should be the last agent's output
        assert result.consensus == "Final response"

    @pytest.mark.asyncio
    async def test_sequential_duration_tracking(self, multiple_agent_configs):
        """Test that execution duration is tracked."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        orchestrator = SequentialOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Test")

        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0


class TestConcurrentOrchestrator:
    """Test concurrent orchestration pattern."""

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, multiple_agent_configs):
        """Test concurrent agent execution."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        orchestrator = ConcurrentOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Test task")

        assert result.pattern == OrchestrationPattern.CONCURRENT
        assert len(result.final_outputs) == 3
        assert result.termination_reason == TerminationReason.MAX_TURNS_REACHED

    @pytest.mark.asyncio
    async def test_concurrent_aggregation(self, multiple_agent_configs):
        """Test that concurrent responses are aggregated."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        orchestrator = ConcurrentOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Test")

        # Consensus should contain all agent names
        assert result.consensus is not None
        assert "Agent 1" in result.consensus
        assert "Agent 2" in result.consensus
        assert "Agent 3" in result.consensus

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, multiple_agent_configs):
        """Test error handling in concurrent execution."""

        class ErrorProvider(MockLLMProvider):
            async def generate(
                self,
                messages: list[Message],
                temperature: float = 0.7,
                max_tokens: int | None = None,
                **kwargs: Any,
            ) -> AgentResponse:
                raise ValueError("Test error")

        error_provider = ErrorProvider()
        good_provider = MockLLMProvider(response="Good response")

        agents = [
            LLMAgent(config=multiple_agent_configs[0], provider=error_provider),
            LLMAgent(config=multiple_agent_configs[1], provider=good_provider),
        ]

        orchestrator = ConcurrentOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Test")

        # Should have one successful output
        assert len(result.final_outputs) == 1

    @pytest.mark.asyncio
    async def test_concurrent_metadata(self, multiple_agent_configs):
        """Test metadata in concurrent results."""
        provider = MockLLMProvider(response="Response")
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        orchestrator = ConcurrentOrchestrator()
        result = await orchestrator.execute(agents=agents, task="Test")

        assert result.metadata["num_agents"] == 3
        assert result.metadata["successful_agents"] == 3
        assert len(result.metadata["agent_names"]) == 3
