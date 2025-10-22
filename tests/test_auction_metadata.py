"""Tests for new auction metadata and provisional interrupts."""

import pytest
from conftest import MockLLMProvider

from argentum.agents.base import AgentConfig
from argentum.agents.llm_agent import LLMAgent
from argentum.audio.controller import SimAudioController
from argentum.coordination.auction_manager import (
    AuctionChatManager,
    EnhancedAuctionResolver,
    SimpleInterruptPolicy,
    SimpleTokenManager,
)
from argentum.memory.context import Context
from argentum.orchestration.auction_chat import AuctionGroupChatOrchestrator


@pytest.mark.asyncio
async def test_provisional_interrupt_event_emitted_when_flag():
    provider = MockLLMProvider(response="Segment content. One. Two.")
    a1 = LLMAgent(config=AgentConfig(name="Alpha", persona="p"), provider=provider)
    a2 = LLMAgent(config=AgentConfig(name="Beta", persona="p"), provider=provider)

    manager = AuctionChatManager(
        token_manager=SimpleTokenManager(max_bank=5),
        emotion_engine=object(),
        resolver=EnhancedAuctionResolver(state={}),
        interrupt_policy=SimpleInterruptPolicy(),
        interjections=object(),
    )
    audio = SimAudioController(words_per_second=1.5, break_ms=100, first_chunk_latency_ms=50)
    orch = AuctionGroupChatOrchestrator(
        manager=manager,
        audio=audio,
        micro_turns=1,
        enable_interjections=False,
        enable_interrupts=True,
        emit_provisional_interrupt=True,
    )
    result = await orch.execute([a1, a2], task="Discuss auction mode.", context=Context())

    assert any((m.metadata or {}).get("event") == "interrupt_provisional" for m in result.messages if m.metadata)


@pytest.mark.asyncio
async def test_segment_metadata_includes_auction_fields():
    provider = MockLLMProvider(response="Segment content. One. Two.")
    a1 = LLMAgent(config=AgentConfig(name="Alpha", persona="p"), provider=provider)
    a2 = LLMAgent(config=AgentConfig(name="Beta", persona="p"), provider=provider)

    manager = AuctionChatManager(
        token_manager=SimpleTokenManager(max_bank=5),
        emotion_engine=object(),
        resolver=EnhancedAuctionResolver(state={}),
        interrupt_policy=SimpleInterruptPolicy(),
        interjections=object(),
    )
    audio = SimAudioController()
    orch = AuctionGroupChatOrchestrator(manager=manager, audio=audio, micro_turns=1, enable_interjections=False)
    result = await orch.execute([a1, a2], task="Check metadata.", context=Context())

    segs = [m for m in result.messages if (m.metadata or {}).get("event") == "segment"]
    assert segs, "no segment messages present"
    meta = segs[0].metadata or {}
    assert "timing" in meta and "auction" in meta
    auct = meta["auction"] or {}
    assert "bids" in auct and "pacing" in auct and "talk_share" in auct


@pytest.mark.asyncio
async def test_prefetch_is_attempted_when_enabled():
    provider = MockLLMProvider(response="Segment content. One. Two.")
    a1 = LLMAgent(config=AgentConfig(name="Alpha", persona="p"), provider=provider)
    a2 = LLMAgent(config=AgentConfig(name="Beta", persona="p"), provider=provider)

    manager = AuctionChatManager(
        token_manager=SimpleTokenManager(max_bank=5),
        emotion_engine=object(),
        resolver=EnhancedAuctionResolver(state={}),
        interrupt_policy=SimpleInterruptPolicy(),
        interjections=object(),
    )
    audio = SimAudioController()
    orch = AuctionGroupChatOrchestrator(
        manager=manager,
        audio=audio,
        micro_turns=2,
        enable_interjections=False,
        enable_interrupts=False,
    )
    _ = await orch.execute([a1, a2], task="Prefetch attempted.", context=Context())

    # Expect exactly two provider calls: first turn + reserve prefetch reused for second
    assert provider.call_count >= 3
