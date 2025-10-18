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

