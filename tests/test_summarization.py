"""Tests for summarization strategies."""

import os
from types import SimpleNamespace

import pytest

from argentum.models import (
    AgentResponse,
    Message,
    MessageType,
    OrchestrationPattern,
    OrchestrationResult,
    TerminationReason,
)
from argentum.workspace.summarization import (
    FrontierSummaryStrategy,
    HeuristicSummaryStrategy,
    LocalSummaryStrategy,
)


def _sample_result() -> OrchestrationResult:
    messages = [
        Message(type=MessageType.USER, sender="orchestrator", content="Discuss recycling."),
        Message(type=MessageType.ASSISTANT, sender="Agent", content="I favour incentives."),
    ]
    outputs = [AgentResponse(agent_name="Agent", content="I favour incentives.")]
    return OrchestrationResult(
        pattern=OrchestrationPattern.GROUP_CHAT,
        messages=messages,
        final_outputs=outputs,
        consensus="Provide incentives for recycling.",
        termination_reason=TerminationReason.CONSENSUS_REACHED,
        metadata={"agent_names": ["Agent"]},
        duration_seconds=12.0,
    )


def test_heuristic_strategy_uses_consensus() -> None:
    strategy = HeuristicSummaryStrategy()
    result = _sample_result()
    summary = strategy.summarize(result, {"topic": "Recycling"}, [])
    assert "Provide incentives" in summary


def test_frontier_strategy_success(monkeypatch) -> None:
    result = _sample_result()

    class DummyCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Frontier summary"))])

    class DummyClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=DummyCompletions())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("argentum.workspace.summarization.openai", SimpleNamespace(OpenAI=lambda **kwargs: DummyClient()))

    strategy = FrontierSummaryStrategy()
    summary = strategy.summarize(result, {"topic": "Recycling"}, [])
    assert summary == "Frontier summary"


def test_frontier_strategy_fallback_when_missing_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    strategy = FrontierSummaryStrategy()
    summary = strategy.summarize(_sample_result(), {"topic": "Recycling"}, [])
    assert "Provide incentives" in summary
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY not set" in captured.out


def test_local_strategy_success(monkeypatch) -> None:
    result = _sample_result()

    class DummyCompletedProcess:
        def __init__(self):
            self.stdout = b"Local summary"

    def fake_run(*args, **kwargs):
        return DummyCompletedProcess()

    monkeypatch.setattr("subprocess.run", fake_run)

    strategy = LocalSummaryStrategy(command=("echo", "test"))
    summary = strategy.summarize(result, {"topic": "Recycling"}, [])
    assert summary == "Local summary"


def test_local_strategy_fallback(monkeypatch, capsys) -> None:
    def fake_run(*args, **kwargs):
        raise RuntimeError("failure")

    monkeypatch.setattr("subprocess.run", fake_run)
    strategy = LocalSummaryStrategy(command=("false",))
    summary = strategy.summarize(_sample_result(), {"topic": "Recycling"}, [])
    assert "Provide incentives" in summary
    captured = capsys.readouterr()
    assert "Local summariser failed" in captured.out


def test_local_strategy_no_command(capsys) -> None:
    strategy = LocalSummaryStrategy(command=())
    summary = strategy.summarize(_sample_result(), {"topic": "Recycling"}, [])
    assert "Provide incentives" in summary
    captured = capsys.readouterr()
    assert "Local summary command not provided" in captured.out
