"""Chat manager for coordinating multi-agent conversations."""

from collections.abc import Sequence
from enum import Enum
from typing import Any

from argentum.agents.base import Agent
from argentum.memory.context import Context
from argentum.models import TerminationReason


class SpeakerSelectionMode(str, Enum):
    """Mode for selecting the next speaker."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    MANUAL = "manual"
    AUTO = "auto"  # LLM decides


class ChatManager:
    """Manages conversation flow in multi-agent group chats."""

    def __init__(
        self,
        max_turns: int = 10,
        selection_mode: SpeakerSelectionMode = SpeakerSelectionMode.ROUND_ROBIN,
        allow_repeats: bool = True,
        min_turns: int = 2,
    ):
        """Initialize the chat manager.

        Args:
            max_turns: Maximum number of conversation turns
            selection_mode: How to select the next speaker
            allow_repeats: Whether agents can speak multiple times
            min_turns: Minimum number of turns before termination
        """
        self.max_turns = max_turns
        self.selection_mode = selection_mode
        self.allow_repeats = allow_repeats
        self.min_turns = min_turns
        self._current_turn = 0
        self._speaker_history: list[str] = []

    def select_next_speaker(
        self,
        agents: Sequence[Agent],
        context: Context,  # noqa: ARG002
    ) -> Agent | None:
        """Select the next speaker in the conversation.

        Args:
            agents: List of available agents
            context: Current conversation context (reserved for future use)

        Returns:
            The selected agent or None if no selection is possible
        """
        if not agents:
            return None

        # Check if max turns reached
        if self._current_turn >= self.max_turns:
            return None

        if self.selection_mode == SpeakerSelectionMode.ROUND_ROBIN:
            return self._select_round_robin(agents)
        else:
            # Default to round-robin for now
            return self._select_round_robin(agents)

    def _select_round_robin(self, agents: Sequence[Agent]) -> Agent:
        """Select next speaker in round-robin fashion.

        Args:
            agents: Available agents

        Returns:
            Next agent
        """
        index = self._current_turn % len(agents)
        return agents[index]

    def should_terminate(
        self,
        context: Context,  # noqa: ARG002
    ) -> tuple[bool, TerminationReason]:
        """Check if conversation should terminate.

        Args:
            context: Current conversation context (reserved for future use)

        Returns:
            Tuple of (should_terminate, reason)
        """
        # Check max turns
        if self._current_turn >= self.max_turns:
            return True, TerminationReason.MAX_TURNS_REACHED

        # Check minimum turns requirement
        if self._current_turn < self.min_turns:
            return False, TerminationReason.MAX_TURNS_REACHED

        # Additional termination checks can be added here
        # e.g., consensus detection, judge decision, etc.

        return False, TerminationReason.MAX_TURNS_REACHED

    def record_turn(self, agent: Agent) -> None:
        """Record that an agent has taken a turn.

        Args:
            agent: Agent that spoke
        """
        self._current_turn += 1
        self._speaker_history.append(agent.name)

    def reset(self) -> None:
        """Reset the chat manager state."""
        self._current_turn = 0
        self._speaker_history.clear()

    def get_turn_count(self) -> int:
        """Get the current turn count.

        Returns:
            Number of turns taken
        """
        return self._current_turn

    def get_speaker_history(self) -> list[str]:
        """Get the history of speakers.

        Returns:
            List of agent names in order of speaking
        """
        return self._speaker_history.copy()

    def get_statistics(self) -> dict[str, Any]:
        """Get conversation statistics.

        Returns:
            Dictionary of statistics
        """
        from collections import Counter

        speaker_counts = Counter(self._speaker_history)

        return {
            "total_turns": self._current_turn,
            "speaker_counts": dict(speaker_counts),
            "unique_speakers": len(speaker_counts),
            "turns_remaining": max(0, self.max_turns - self._current_turn),
        }
