"""Auction manager composition and interfaces.

This module defines lightweight interfaces/stubs for the sub-managers used by
the auction-based orchestrator. Implementations will be filled in iteratively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class AgentAuctionState:
    tokens: int = 0
    frustration: float = 0.0
    engagement: float = 0.5
    confidence: float = 0.5
    reputation: float = 0.0
    interrupts_made: int = 0
    interjections_made: int = 0


class TokenManager(Protocol):  # pragma: no cover - interface
    def accrue(self, state: dict[str, AgentAuctionState]) -> None: ...
    def charge(self, agent: str, amount: int) -> None: ...


class EmotionEngine(Protocol):  # pragma: no cover - interface
    def update_from_llm(self, agent: str, payload: dict[str, float]) -> None: ...
    def nudge_from_event(self, agent: str, event: str) -> None: ...


class AuctionResolver(Protocol):  # pragma: no cover - interface
    async def collect_bids(self, agents: list[str], context: dict[str, Any]) -> dict[str, dict[str, Any]]: ...
    def resolve(self, bids: dict[str, dict[str, Any]]) -> tuple[str | None, bool]: ...


class InterruptPolicy(Protocol):  # pragma: no cover - interface
    def can_interrupt(self, agent: str) -> bool: ...
    def record_interrupt(self, agent: str) -> None: ...


class InterjectionPlanner(Protocol):  # pragma: no cover - interface
    async def plan(self, agents: list[str], context: dict[str, Any]) -> list[dict[str, Any]]: ...


@dataclass
class AuctionChatManager:
    token_manager: TokenManager
    emotion_engine: EmotionEngine
    resolver: AuctionResolver
    interrupt_policy: InterruptPolicy
    interjections: InterjectionPlanner
    state: dict[str, AgentAuctionState] = field(default_factory=dict)

    def get_or_create(self, agent: str) -> AgentAuctionState:
        if agent not in self.state:
            self.state[agent] = AgentAuctionState()
        return self.state[agent]


# ------------------------- simple implementations ---------------------------

def _softmax(scores: list[float], tau: float = 0.3) -> list[float]:
    import math

    if not scores:
        return []
    # temperature-scaled softmax
    mx = max(scores)
    exps = [math.exp((s - mx) / max(1e-6, tau)) for s in scores]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


@dataclass
class SimpleTokenManager:
    """Per-segment accrual with a max bank and safe charge."""

    max_bank: int = 8

    def accrue(self, state: dict[str, AgentAuctionState]) -> None:
        for st in state.values():
            if st.tokens < self.max_bank:
                st.tokens += 1

    def charge(self, agent: str, amount: int) -> None:
        # Charging will be executed by the caller with a validated agent/state
        _ = agent, amount


@dataclass
class SimpleInterruptPolicy:
    """Sliding window-free interrupt budget per agent (simplified)."""

    max_interrupts_per_window: int = 2

    def can_interrupt(self, agent: str) -> bool:
        return True

    def record_interrupt(self, agent: str) -> None:
        _ = agent


@dataclass
class SimpleAuctionResolver:
    """Heuristic resolver with pacing and softmax bid mapping.

    This simplified implementation reads AgentAuctionState and computes a
    desire score from tokens and emotion, then maps to an integer bid using a
    softmax over all desires. Pacing is not persisted in this version.
    """

    state: dict[str, AgentAuctionState]
    max_bid: int = 5

    async def collect_bids(self, agents: list[str], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
        desires: list[float] = []
        ordering: list[str] = []
        for name in agents:
            st = self.state.get(name) or AgentAuctionState()
            # Simple desire: tokens plus a small boost from frustration
            desire = st.tokens + 0.5 * max(0.0, st.frustration - 0.5)
            desires.append(desire)
            ordering.append(name)
        probs = _softmax(desires, tau=0.3) if desires else []
        bids: dict[str, dict[str, Any]] = {}
        for name, p in zip(ordering, probs):
            st = self.state.get(name) or AgentAuctionState()
            # Scale by both max_bid and available tokens
            suggested = int(round(p * self.max_bid))
            amount = max(0, min(st.tokens, suggested))
            bids[name] = {"amount": amount, "interrupt": False}
        return bids

    def resolve(self, bids: dict[str, dict[str, Any]]) -> tuple[str | None, bool]:
        if not bids:
            return None, False
        # Choose highest amount; tie-break by name
        winner = max(sorted(bids.items(), key=lambda kv: kv[0]), key=lambda kv: kv[1].get("amount", 0))
        agent_name = winner[0]
        return agent_name, bool(winner[1].get("interrupt"))


# ------------------------- enhanced implementations --------------------------

@dataclass
class WindowInterruptPolicy:
    """Interrupt budget over a sliding window of segments per agent.

    Call `start_new_segment()` at the beginning of each segment. An agent can
    only record up to `max_per_window` hard interrupts within the last
    `window_segments`. This policy is time-agnostic and segment-based.
    """

    max_per_window: int = 2
    window_segments: int = 5
    _history: dict[str, list[int]] = field(default_factory=dict)

    def start_new_segment(self) -> None:
        # Append a 0 slot for current segment for all agents seen so far
        for key, slots in self._history.items():
            slots.append(0)
            if len(slots) > self.window_segments:
                del slots[0]

    def can_interrupt(self, agent: str) -> bool:
        slots = self._history.get(agent)
        if not slots:
            return True
        return sum(slots) < self.max_per_window

    def record_interrupt(self, agent: str) -> None:
        slots = self._history.setdefault(agent, [])
        # Ensure current segment slot exists
        if not slots or (len(slots) > 0 and (len(slots) == 0)):
            slots.append(0)
        if not slots:
            slots.append(0)
        # Increment current segment's interrupt count
        slots[-1] = slots[-1] + 1
        if len(slots) > self.window_segments:
            del slots[0]


@dataclass
class EnhancedAuctionResolver:
    """Resolver with pacing multiplier, softmax bid mapping, and evidence tie-break.

    - Pacing: per-agent multiplier p_i ∈ [0.5, 1.0] updated from talk-share
      versus target_share using EMA with smoothing factor `alpha`.
    - Softmax: map desires to a probability distribution, then to suggested
      integer bids scaled by max_bid and capped by available tokens; apply pacing
      multiplier to the suggested bid.
    - Evidence priority: when resolving ties on amount, prefer evidence-backed bids.
    """

    state: dict[str, AgentAuctionState]
    max_bid: int = 5
    tau: float = 0.3
    alpha: float = 0.05
    target_share: float | None = None
    pacing: dict[str, float] = field(default_factory=dict)

    def update_pacing(self, talk_share: dict[str, float], *, target_share: float | None = None) -> None:
        tgt = target_share or self.target_share or (1.0 / max(1, len(talk_share) or 1))
        for agent, share in talk_share.items():
            p_prev = self.pacing.get(agent, 1.0)
            excess = max(0.0, share - tgt)
            # EMA: push p toward (1 - excess) within [0.5, 1.0]
            p_new = (1 - self.alpha) * p_prev + self.alpha * (1.0 - excess)
            self.pacing[agent] = min(1.0, max(0.5, p_new))

    async def collect_bids(self, agents: list[str], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
        desires: list[float] = []
        ordering: list[str] = []
        for name in agents:
            st = self.state.get(name) or AgentAuctionState()
            desire = st.tokens + 0.5 * max(0.0, st.frustration - 0.5)
            desires.append(desire)
            ordering.append(name)
        probs = _softmax(desires, tau=self.tau) if desires else []
        evidence_map = context.get("evidence") or {}
        bids: dict[str, dict[str, Any]] = {}
        for name, p in zip(ordering, probs):
            st = self.state.get(name) or AgentAuctionState()
            suggested = int(round(p * self.max_bid))
            paced = int(round(self.pacing.get(name, 1.0) * suggested))
            amount = max(0, min(st.tokens, paced))
            evidence = bool(evidence_map.get(name))
            bids[name] = {"amount": amount, "interrupt": False, "evidence": evidence}
        return bids

    def resolve(self, bids: dict[str, dict[str, Any]]) -> tuple[str | None, bool]:
        if not bids:
            return None, False
        # Group by amount, then prefer evidence-backed on ties, then alphabetical
        max_amount = max(bid.get("amount", 0) for bid in bids.values())
        candidates = [name for name, bid in bids.items() if bid.get("amount", 0) == max_amount]
        if not candidates:
            return None, False
        # Prefer evidence-backed among candidates
        evidence_candidates = [name for name in candidates if bids[name].get("evidence")]
        pool = evidence_candidates or candidates
        winner = sorted(pool)[0]
        return winner, bool(bids[winner].get("interrupt"))
