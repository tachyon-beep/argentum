"""Unit tests for LLMAgent retrieval flow."""

import asyncio
from typing import Any

import pytest

from argentum.agents.base import AgentConfig
from argentum.agents.llm_agent import LLMAgent
from argentum.llm.provider import LLMProvider
from argentum.models import Message, MessageType, Role


class DummyProvider(LLMProvider):
    """Deterministic provider that triggers a retrieval round-trip."""

    def __init__(self) -> None:
        self.calls = 0

    def count_tokens(self, messages: list[Message]) -> int:
        return sum(len(msg.content) for msg in messages)

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        if self.calls == 0:
            self.calls += 1
            return "<<retrieve: market trends>>"
        return "After reviewing [Doc 1], we should proceed."

    async def generate_with_tools(self, messages: list[Message], tools: list[dict[str, Any]], **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_model_name(self) -> str:
        return "dummy"


class FakeRetriever:
    """Simple retriever stub that returns a single labeled document."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def register_documents(self, documents: list[dict[str, Any]]) -> None:  # noqa: D401 - compatibility
        """No-op for initial registration."""

    def search(self, query: str, *, limit: int | None = None, min_score: float | None = None) -> list[dict[str, Any]]:
        self.history.append({"query": query, "labels": ["Doc 1"], "count": 1})
        return [
            {
                "doc_id": "policy",
                "chunk_id": "policy::chunk-1",
                "text": "Market trends indicate consistent growth.",
                "metadata": {"document": {"title": "Market Outlook"}},
                "label": "Doc 1",
            }
        ]


@pytest.mark.asyncio
async def test_llm_agent_mid_session_retrieval() -> None:
    provider = DummyProvider()
    retriever = FakeRetriever()

    agent = LLMAgent(
        config=AgentConfig(
            name="Advisor",
            role=Role.ADVISOR,
            persona="Provide decisive advice.",
            model="dummy",
            temperature=0.0,
            max_tokens=100,
        ),
        provider=provider,
        retriever=retriever,
    )

    context: dict[str, Any] = {
        "question": "Should we scale production?",
        "knowledge_documents": [],
    }

    response = await agent.generate_response(messages=[], context=context)

    assert "Doc 1" in response.content
    assert response.metadata.get("citations")
    assert retriever.history
    assert context["knowledge_documents"][0]["label"] == "Doc 1"
    # Retrieval directive should be stripped from the final response
    assert "<<retrieve" not in response.content.lower()
