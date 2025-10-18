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


class SimPlaybackHandle(PlaybackHandle):
    """Playback handle that simulates timing and beat marks.

    This implementation does not render audio; it tracks a schedule of beat
    timestamps and allows callers to await the end of a given beat.
    """

    def __init__(self, id: str, beat_times: list[float], loop: asyncio.AbstractEventLoop | None = None) -> None:
        super().__init__(id=id)
        self._loop = loop or asyncio.get_event_loop()
        self._beat_times = beat_times
        self._cancelled = False
        self._finished = asyncio.Event()
        self._beat_events: list[asyncio.Event] = [asyncio.Event() for _ in beat_times]
        self._task = self._loop.create_task(self._runner())

    async def _runner(self) -> None:
        start = self._loop.time()
        for idx, t_rel in enumerate(self._beat_times):
            if self._cancelled:
                break
            # Sleep until the scheduled relative time
            now = self._loop.time()
            delay = max(0.0, (start + t_rel) - now)
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:  # pragma: no cover - defensive
                break
            self._beat_events[idx].set()
        self._finished.set()

    def can_interrupt(self) -> bool:
        return not self._cancelled and not self._finished.is_set()

    async def wait_for_beat(self, beat_index: int) -> None:
        # 1-based indexing support
        idx = beat_index - 1 if beat_index > 0 else beat_index
        if 0 <= idx < len(self._beat_events):
            await self._beat_events[idx].wait()

    async def finish(self) -> None:
        await self._finished.wait()

    def cancel(self) -> None:
        self._cancelled = True
        if not self._task.done():
            self._task.cancel()


class SimAudioController(AudioController):
    """A simple simulated audio controller with estimated beat timing.

    Beats are computed by splitting the input into sentences and mapping word
    counts to durations using an estimated words-per-second rate.
    """

    def __init__(
        self,
        *,
        words_per_second: float = 2.7,
        break_ms: int = 250,
        first_chunk_latency_ms: int = 200,
    ) -> None:
        self._wps = max(0.5, words_per_second)
        self._break_ms = max(0, break_ms)
        self._first_ms = max(0, first_chunk_latency_ms)
        self._active: SimPlaybackHandle | None = None
        self._loop = asyncio.get_event_loop()

    async def play(self, ssml_or_text: str, *, clip_id: str | None = None) -> PlaybackHandle:
        _ = clip_id
        # Cancel any current playback
        if self._active is not None:
            self._active.cancel()

        sentences = _split_sentences(ssml_or_text)
        rel_times: list[float] = []
        t = self._first_ms / 1000.0
        for sent in sentences:
            words = max(1, _count_words(sent))
            dur = words / self._wps
            t += dur
            rel_times.append(t)
            t += self._break_ms / 1000.0

        handle = SimPlaybackHandle(id="sim", beat_times=rel_times, loop=self._loop)
        self._active = handle
        return handle

    async def duck_and_play(self, text: str, *, duration_ms: int | None = None) -> None:
        _ = text
        await asyncio.sleep((duration_ms or 600) / 1000.0)

    async def crossfade_to_silence(self, duration_ms: int = 200) -> None:
        await asyncio.sleep(duration_ms / 1000.0)
        if self._active is not None:
            self._active.cancel()
            self._active = None

    async def cancel_all(self) -> None:
        if self._active is not None:
            self._active.cancel()
            self._active = None


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _count_words(text: str) -> int:
    return len([t for t in text.strip().split() if t])
