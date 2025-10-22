"""Tests the hard-interrupt cutoff path in the orchestrator (simulated)."""

import pytest
from conftest import MockLLMProvider

from argentum.agents.base import AgentConfig
from argentum.agents.llm_agent import LLMAgent
from argentum.audio.controller import SimAudioController
from argentum.coordination.auction_manager import (
    AuctionChatManager,
    SimpleAuctionResolver,
    SimpleInterruptPolicy,
    SimpleTokenManager,
)
from argentum.memory.context import Context
from argentum.orchestration.auction_chat import AuctionGroupChatOrchestrator


@pytest.mark.asyncio
async def test_hard_interrupt_event_emitted_when_enabled():
    # Provider returns simple predictable content with multiple sentences
    provider = MockLLMProvider(response="Segment content. One. Two.")
    a1 = LLMAgent(config=AgentConfig(name="Agent A", persona="p"), provider=provider)
    a2 = LLMAgent(config=AgentConfig(name="Agent B", persona="p"), provider=provider)

    manager = AuctionChatManager(
        token_manager=SimpleTokenManager(max_bank=5),
        emotion_engine=object(),
        resolver=SimpleAuctionResolver(state={}),
        interrupt_policy=SimpleInterruptPolicy(),
        interjections=object(),
    )
    audio = SimAudioController(words_per_second=1.0, break_ms=100, first_chunk_latency_ms=50)
    orch = AuctionGroupChatOrchestrator(
        manager=manager, audio=audio, micro_turns=1, enable_interjections=False, enable_interrupts=True
    )
    result = await orch.execute([a1, a2], task="Discuss auction mode.", context=Context())

    # Expect a hard interrupt event to be present
    assert any(
        (m.metadata or {}).get("event") == "hard_interrupt" for m in result.messages if m.metadata
    )
