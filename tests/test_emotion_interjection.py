"""Tests for SimpleEmotionEngine and BasicInterjectionPlanner."""

import pytest

from argentum.coordination.auction_manager import (
    AgentAuctionState,
    BasicInterjectionPlanner,
    SimpleEmotionEngine,
)


def test_emotion_engine_smoothing_and_nudges():
    state = {"alice": AgentAuctionState(tokens=0)}
    eng = SimpleEmotionEngine(state=state, alpha=0.5, deadband=0.05, nudge_step=0.2, mode="llm", cap_per_turn=0.3, min_dwell_turns=2)
    # Initial values default to 0/0.5/0.5
    eng.update_from_llm("alice", {"frustration": 0.8, "engagement": 0.2, "confidence": 0.9})
    s = state["alice"]
    # EMA 0.5: f=0.4, e=0.35, c=0.7 (with initial 0/0.5/0.5)
    assert 0.30 <= s.frustration <= 0.45
    assert 0.3 <= s.engagement <= 0.4
    assert 0.65 <= s.confidence <= 0.75
    # Hysteresis: immediate flip should be damped
    prev_f = s.frustration
    eng.update_from_llm("alice", {"frustration": 0.1})  # large drop request
    s2 = state["alice"]
    # With dwell=2 and cap, change should not exceed cap and may be deferred
    assert s2.frustration >= prev_f - 0.3

    # Nudge events
    eng.nudge_from_event("alice", "lost_bid")
    assert s.frustration >= 0.5  # increased
    eng.nudge_from_event("alice", "won_interrupt")
    assert s.confidence >= 0.7  # rebounds


@pytest.mark.asyncio
async def test_basic_interjection_planner_cooldown():
    planner = BasicInterjectionPlanner(cooldown_segments=1)
    plan1 = await planner.plan(["bob"], context={"type": "clarify"})
    assert plan1 and plan1[0]["type"] == "clarify" and 0.0 < plan1[0]["importance"] <= 1.0
    # Immediately planning again should respect cooldown → empty
    plan2 = await planner.plan(["bob"], context={})
    assert plan2 == []
    # Advance segment window: cooldown clears
    planner.start_new_segment()
    plan3 = await planner.plan(["bob"], context={})
    assert plan3 and plan3[0]["text"]
