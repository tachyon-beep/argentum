"""Tests for enhanced auction resolver and windowed interrupt policy."""

import pytest

from argentum.coordination.auction_manager import (
    AgentAuctionState,
    EnhancedAuctionResolver,
    WindowInterruptPolicy,
)


def test_window_interrupt_policy_budget():
    pol = WindowInterruptPolicy(max_per_window=2, window_segments=5)
    # Start first segment
    pol.start_new_segment()
    assert pol.can_interrupt("a") is True
    pol.record_interrupt("a")
    assert pol.can_interrupt("a") is True
    pol.record_interrupt("a")
    # Now budget exhausted for the window
    assert pol.can_interrupt("a") is False
    # Advance window segments; after 5 segments budget should free
    for _ in range(5):
        pol.start_new_segment()
    assert pol.can_interrupt("a") is True


@pytest.mark.asyncio
async def test_enhanced_resolver_pacing_and_evidence_tiebreak():
    # Two agents with same tokens; alpha has excess talk share
    state = {
        "alpha": AgentAuctionState(tokens=4, frustration=0.3),
        "beta": AgentAuctionState(tokens=4, frustration=0.3),
    }
    resolver = EnhancedAuctionResolver(state=state, max_bid=4, tau=0.3, alpha=0.1)
    resolver.update_pacing({"alpha": 0.6, "beta": 0.4}, target_share=0.5)
    bids = await resolver.collect_bids(["alpha", "beta"], context={})
    # Expect alpha's bid to be <= beta's due to pacing multiplier < 1
    assert bids["alpha"]["amount"] <= bids["beta"]["amount"]

    # Force a tie on amount but give beta evidence-backed flag
    tied = {"alpha": {"amount": 2, "evidence": False}, "beta": {"amount": 2, "evidence": True}}
    winner, interrupt = resolver.resolve(tied)
    assert winner == "beta"
    assert interrupt is False

