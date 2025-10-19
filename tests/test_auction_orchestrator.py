"""Basic tests for the auction orchestrator micro-turn loop."""

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
from argentum.models import OrchestrationPattern
from argentum.orchestration.auction_chat import AuctionGroupChatOrchestrator


@pytest.mark.asyncio
async def test_auction_orchestrator_runs_micro_turns():
    provider = MockLLMProvider(response="Segment content. One. Two.")
    cfg1 = AgentConfig(name="Agent A", persona="p")
    cfg2 = AgentConfig(name="Agent B", persona="p")
    a1 = LLMAgent(config=cfg1, provider=provider)
    a2 = LLMAgent(config=cfg2, provider=provider)

    manager = AuctionChatManager(
        token_manager=SimpleTokenManager(max_bank=5),
        emotion_engine=object(),  # not used in simple flow
        resolver=SimpleAuctionResolver(state={}),
        interrupt_policy=SimpleInterruptPolicy(),
        interjections=object(),  # not used; orchestrator synthesizes basic interjection
    )
    audio = SimAudioController()
    orch = AuctionGroupChatOrchestrator(manager=manager, audio=audio, micro_turns=2)
    result = await orch.execute([a1, a2], task="Discuss auction mode.", context=Context())

    assert result.pattern == OrchestrationPattern.GROUP_CHAT
    assert any(msg.metadata.get("event") == "interjection" for msg in result.messages if msg.metadata)
    assert any(msg.metadata.get("event") == "segment" for msg in result.messages if msg.metadata)

