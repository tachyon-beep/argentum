"""Abstract audio controller for TTS playback and mixing.

This module defines a minimal async interface used by the auction orchestrator
to coordinate playback, crossfades, ducking, beat waits, and cancellation.

Concrete implementations may wrap streaming TTS providers such as ElevenLabs.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PlaybackHandle:
    """Represents a single in-flight playback.

    Implementations should ensure methods are idempotent and safe under
    cancellation. `wait_for_beat` should resolve on provider marks if
    available; otherwise approximate using schedule offsets.
    """

    id: str

    async def wait_for_beat(self, beat_index: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def can_interrupt(self) -> bool:  # pragma: no cover - interface
        return True

    async def finish(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class AudioController(ABC):
    """Abstract controller for audio playback and transitions."""

    @abstractmethod
    async def play(self, ssml_or_text: str, *, clip_id: str | None = None) -> PlaybackHandle:
        """Start playback of a segment. Returns a handle used to await beats or stop.

        Args:
            ssml_or_text: Content to synthesize and play.
            clip_id: Optional stable identifier for observability.
        """

    @abstractmethod
    async def duck_and_play(self, text: str, *, duration_ms: int | None = None) -> None:
        """Briefly duck current output and play a short interjection clip."""

    @abstractmethod
    async def crossfade_to_silence(self, duration_ms: int = 200) -> None:
        """Fade out the current clip to silence over the given duration."""

    @abstractmethod
    async def cancel_all(self) -> None:
        """Stop all audio and release resources."""


class NoOpPlaybackHandle(PlaybackHandle):
    def __init__(self, id: str = "noop") -> None:
        super().__init__(id=id)
        self._event = asyncio.Event()

    async def wait_for_beat(self, beat_index: int) -> None:  # pragma: no cover - trivial
        _ = beat_index
        # Simulate a short wait to preserve sequencing in tests/demos
        await asyncio.sleep(0.01)

    async def finish(self) -> None:  # pragma: no cover - trivial
        self._event.set()


class NoOpAudioController(AudioController):
    """A minimal controller that simulates timing without producing audio."""

    async def play(self, ssml_or_text: str, *, clip_id: str | None = None) -> PlaybackHandle:
        _ = ssml_or_text, clip_id
        return NoOpPlaybackHandle()

    async def duck_and_play(self, text: str, *, duration_ms: int | None = None) -> None:
        _ = text, duration_ms
        await asyncio.sleep(0)

    async def crossfade_to_silence(self, duration_ms: int = 200) -> None:
        _ = duration_ms
        await asyncio.sleep(0)

    async def cancel_all(self) -> None:
        await asyncio.sleep(0)

