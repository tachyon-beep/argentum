"""Readiness/fallback behavior for hard-interrupt cutoff."""

import asyncio
import pytest

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


class SlowMockProvider:
    def __init__(self, response: str = "Slow response", delay_s: float = 0.35) -> None:
        self.response = response
        self.delay_s = delay_s

    async def generate(self, messages, _temperature: float = 0.7, _max_tokens: int = 1000, **kwargs):
        await asyncio.sleep(self.delay_s)
        return self.response

    def get_model_name(self) -> str:
        return "slow-mock"

    def count_tokens(self, messages):
        return 42


class FastMockProvider(SlowMockProvider):
    def __init__(self, response: str = "Fast response") -> None:
        super().__init__(response=response, delay_s=0.0)


@pytest.mark.asyncio
async def test_interrupt_readiness_miss_falls_back():
    # Speaker (Agent A) is fast; candidate interrupter (Agent B) is slow.
    a1 = LLMAgent(config=AgentConfig(name="Agent A", persona="p"), provider=FastMockProvider("One. Two."))
    a2 = LLMAgent(config=AgentConfig(name="Agent B", persona="p"), provider=SlowMockProvider("Kicker", delay_s=0.5))

    manager = AuctionChatManager(
        token_manager=SimpleTokenManager(max_bank=5),
        emotion_engine=object(),
        resolver=SimpleAuctionResolver(state={}),
        interrupt_policy=SimpleInterruptPolicy(),
        interjections=object(),
    )
    audio = SimAudioController()
    orch = AuctionGroupChatOrchestrator(
        manager=manager, audio=audio, micro_turns=1, enable_interjections=False, enable_interrupts=True
    )
    result = await orch.execute([a1, a2], task="Discuss.", context=Context())

    # Since B is too slow vs default guard, interrupt should miss; expect no hard_interrupt events
    assert not any((m.metadata or {}).get("event") == "hard_interrupt" for m in result.messages if m.metadata)
