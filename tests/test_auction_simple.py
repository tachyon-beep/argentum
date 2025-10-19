"""Tests for simple auction components and simulated audio controller."""

import asyncio

import pytest

from argentum.audio.controller import SimAudioController
from argentum.coordination.auction_manager import (
    AgentAuctionState,
    SimpleAuctionResolver,
    SimpleInterruptPolicy,
    SimpleTokenManager,
)


@pytest.mark.asyncio
async def test_sim_audio_controller_beats():
    audio = SimAudioController(words_per_second=3.0, break_ms=100, first_chunk_latency_ms=100)
    handle = await audio.play("Hello there. This is a test.")
    # Should be able to await first beat without error
    await handle.wait_for_beat(1)
    # Finish should eventually resolve
    await handle.finish()


def test_token_manager_accrue_and_charge():
    tm = SimpleTokenManager(max_bank=3)
    state = {
        "a": AgentAuctionState(tokens=2),
        "b": AgentAuctionState(tokens=3),
    }
    tm.accrue(state)
    assert state["a"].tokens == 3  # incremented
    assert state["b"].tokens == 3  # capped at max_bank


@pytest.mark.asyncio
async def test_simple_auction_resolver_softmax_mapping():
    state = {
        "alpha": AgentAuctionState(tokens=5, frustration=0.2),
        "beta": AgentAuctionState(tokens=3, frustration=0.7),
        "gamma": AgentAuctionState(tokens=1, frustration=0.9),
    }
    resolver = SimpleAuctionResolver(state=state, max_bid=5)
    bids = await resolver.collect_bids(["alpha", "beta", "gamma"], context={})
    # Bids should be non-negative and not exceed tokens
    for agent, bid in bids.items():
        assert 0 <= bid["amount"] <= state[agent].tokens
    # Resolve picks the highest bid (tie-break alphabetically by name)
    winner, interrupt = resolver.resolve(bids)
    assert winner in bids
    assert interrupt is False


def test_interrupt_policy_allows_and_records():
    pol = SimpleInterruptPolicy(max_interrupts_per_window=2)
    assert pol.can_interrupt("alpha") is True
    pol.record_interrupt("alpha")
    # Policy is simplified and always allows in this stub
    assert pol.can_interrupt("alpha") is True

