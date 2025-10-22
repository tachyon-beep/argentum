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


# ------------------------- emotion engine & interjections ---------------------

from dataclasses import dataclass

@dataclass
class SimpleEmotionEngine:
    """Emotion engine with EMA smoothing, deadband, hysteresis, and bounded nudges.

    Modes:
      - heuristic: ignore LLM updates; only event nudges apply
      - llm: apply LLM updates
      - hybrid: apply LLM updates with conservative caps
    """

    state: dict[str, AgentAuctionState]
    alpha: float = 0.25
    deadband: float = 0.1
    nudge_step: float = 0.12
    mode: str = "hybrid"  # heuristic|llm|hybrid
    cap_per_turn: float = 0.3  # max absolute change per update
    min_dwell_turns: int = 2
    _dwell: dict[str, tuple[int,int]] = field(default_factory=dict)  # agent -> (dir,count)

    def _smooth(self, prev: float, new: float, agent: str, key: str) -> float:
        if self.mode == "heuristic":
            return prev
        if abs(new - prev) < self.deadband:
            return prev
        dirn = 1 if new > prev else -1
        last = self._dwell.get(f"{agent}:{key}")
        if last is None:
            self._dwell[f"{agent}:{key}"] = (dirn, 1)
        else:
            last_dir, cnt = last
            if dirn != last_dir and cnt < self.min_dwell_turns:
                self._dwell[f"{agent}:{key}"] = (last_dir, cnt + 1)
                return prev
            self._dwell[f"{agent}:{key}"] = (dirn, 1 if dirn != last_dir else cnt + 1)
        a = max(0.0, min(1.0, self.alpha))
        val = (1 - a) * prev + a * new
        lo, hi = prev - self.cap_per_turn, prev + self.cap_per_turn
        val = max(lo, min(hi, val))
        return max(0.0, min(1.0, val))

    def update_from_llm(self, agent: str, payload: dict[str, float]) -> None:
        if self.mode not in {"hybrid", "llm"}:
            return
        st = self.state.setdefault(agent, AgentAuctionState())
        f = float(payload.get("frustration", st.frustration))
        e = float(payload.get("engagement", st.engagement))
        c = float(payload.get("confidence", st.confidence))
        st.frustration = self._smooth(st.frustration, max(0.0, min(1.0, f)), agent, "frustration")
        st.engagement = self._smooth(st.engagement, max(0.0, min(1.0, e)), agent, "engagement")
        st.confidence = self._smooth(st.confidence, max(0.0, min(1.0, c)), agent, "confidence")

    def nudge_from_event(self, agent: str, event: str) -> None:
        st = self.state.setdefault(agent, AgentAuctionState())
        d = self.nudge_step
        if event in {"lost_bid", "was_interrupted"}:
            st.frustration = max(0.0, min(1.0, st.frustration + d))
            st.confidence = max(0.0, min(1.0, st.confidence - 0.5 * d))
        elif event in {"won_bid", "won_interrupt", "spoke_segment"}:
            st.engagement = max(0.0, min(1.0, st.engagement + 0.5 * d))
            st.confidence = max(0.0, min(1.0, st.confidence + 0.5 * d))


@dataclass
class BasicInterjectionPlanner:

    """Typed interjection planner with a simple per-segment cooldown.

    Call `start_new_segment()` to advance the segment window and clear cooldowns.
    The `plan` API returns at most one short interjection for the first eligible
    agent, with a type hint in the result.
    """

    cooldown_segments: int = 1
    _cool: dict[str, int] = field(default_factory=dict)

    def start_new_segment(self) -> None:
        # Decrease cooldown counters; drop expired entries
        drop: list[str] = []
        for k, v in self._cool.items():
            nv = max(0, v - 1)
            if nv == 0:
                drop.append(k)
            else:
                self._cool[k] = nv
        for k in drop:
            self._cool.pop(k, None)

    async def plan(self, agents: list[str], context: dict[str, Any]) -> list[dict[str, Any]]:
        for name in agents:
            if self._cool.get(name):
                continue
            # Pick a type based on a hint in context or default to "challenge"
            itype = (context.get("type") if isinstance(context, dict) else None) or "challenge"
            text = self._template_for_type(itype)
            self._cool[name] = max(1, self.cooldown_segments)
            return [{"agent": name, "type": itype, "text": text, "importance": self._importance(itype)}]
        return []

    def _template_for_type(self, itype: str) -> str:
        t = itype.lower()
        if t == "support":
            return "Good point. I agree with that."
        if t == "clarify":
            return "Can you clarify what you mean?"
        # default: challenge
        return "Why do you think that?"

    def _importance(self, itype: str) -> float:
        t = itype.lower()
        if t == "support":
            return 0.4
        if t == "clarify":
            return 0.6
        return 0.8


# ------------------------- factory helpers -----------------------------------

def create_default_auction_manager(config: dict[str, Any] | None = None) -> AuctionChatManager:
    """Factory for AuctionChatManager with simple components.

    Config keys (optional):
    - emotion_control: "heuristic"|"llm"|"hybrid"
    - interjections.cooldown_segments: int
    - tokens.max_bank: int
    """
    cfg = config or {}
    max_bank = int(((cfg.get("tokens") or {}).get("max_bank") or 8))
    cooldown = int(((cfg.get("interjections") or {}).get("cooldown_segments") or 1))
    emo_mode = str(cfg.get("emotion_control") or "hybrid")

    state: dict[str, AgentAuctionState] = {}
    token_mgr = SimpleTokenManager(max_bank=max_bank)
    emotion = SimpleEmotionEngine(state=state, mode=emo_mode)
    resolver = EnhancedAuctionResolver(state=state)
    interrupt_pol = WindowInterruptPolicy()
    interj = BasicInterjectionPlanner(cooldown_segments=cooldown)
    return AuctionChatManager(
        token_manager=token_mgr,
        emotion_engine=emotion,
        resolver=resolver,
        interrupt_policy=interrupt_pol,
        interjections=interj,
        state=state,
    )
