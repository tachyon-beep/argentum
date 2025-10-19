"""Audio controller backed by a (simulated) ElevenLabs streaming adapter.

This implementation consumes an adapter that yields streaming events including
optional timing marks. When marks are available, they are used to signal beat
boundaries. Otherwise it falls back to estimated beat timings using sentence
lengths and a voice profile.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

from .controller import AudioController, PlaybackHandle


@dataclass(slots=True)
class ElevenLabsVoiceProfile:
    words_per_second: float = 2.7
    break_ms: int = 250
    first_chunk_latency_ms: int = 200


class MarksPlaybackHandle(PlaybackHandle):
    def __init__(self, id: str, beat_count: int, loop: asyncio.AbstractEventLoop | None = None) -> None:
        super().__init__(id=id)
        self._loop = loop or asyncio.get_event_loop()
        self._finished = asyncio.Event()
        self._cancelled = False
        self._beats = [asyncio.Event() for _ in range(max(0, beat_count))]
        self._beat_times_ms: list[int] = [0 for _ in range(max(0, beat_count))]
        self._planned_ms: list[int] = [0 for _ in range(max(0, beat_count))]
        self._start_ms: int = int(self._loop.time() * 1000)

    async def wait_for_beat(self, beat_index: int) -> None:
        idx = beat_index - 1 if beat_index > 0 else beat_index
        if 0 <= idx < len(self._beats):
            await self._beats[idx].wait()

    def set_beat(self, beat_index: int) -> None:
        idx = beat_index - 1 if beat_index > 0 else beat_index
        if 0 <= idx < len(self._beats):
            self._beats[idx].set()
            # Timestamp in ms since loop start
            self._beat_times_ms[idx] = int(self._loop.time() * 1000)

    def set_planned(self, beat_index: int, planned_offset_ms: int) -> None:
        idx = beat_index - 1 if beat_index > 0 else beat_index
        if 0 <= idx < len(self._planned_ms):
            self._planned_ms[idx] = self._start_ms + max(0, planned_offset_ms)

    def drift_ms(self) -> list[int]:
        """Return per-beat drift = actual - planned (ms), 0 if unknown."""
        drift: list[int] = []
        for a, p in zip(self._beat_times_ms, self._planned_ms):
            if a and p:
                drift.append(a - p)
            else:
                drift.append(0)
        return drift

    def can_interrupt(self) -> bool:
        return not self._cancelled and not self._finished.is_set()

    async def finish(self) -> None:
        await self._finished.wait()

    def mark_finished(self) -> None:
        self._finished.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._finished.set()


class ElevenLabsAudioController(AudioController):
    def __init__(
        self,
        adapter: "StreamingAdapter",
        *,
        profile: ElevenLabsVoiceProfile | None = None,
    ) -> None:
        self._adapter = adapter
        self._profile = profile or ElevenLabsVoiceProfile()
        self._active: MarksPlaybackHandle | None = None
        self._loop = asyncio.get_event_loop()
        # single ducking lease
        self._duck_task: asyncio.Task | None = None
        self._duck_log: list[tuple[str, float]] = []  # ("start"|"end", loop_time)
        self._latency_profiler = _LatencyProfiler()

    async def play(self, ssml_or_text: str, *, clip_id: str | None = None) -> PlaybackHandle:
        # Cancel current playback (if any)
        await self.cancel_all()

        sentences = _split_sentences(ssml_or_text)
        handle = MarksPlaybackHandle(id=clip_id or "elevenlabs", beat_count=len(sentences), loop=self._loop)
        self._active = handle

        # Launch a background task to consume stream and emit beat marks
        self._loop.create_task(self._run_stream(ssml_or_text, sentences, handle))
        return handle

    async def _run_stream(self, text: str, sentences: list[str], handle: MarksPlaybackHandle) -> None:
        # If adapter yields explicit mark events, use them; otherwise fall back to estimates
        got_mark = False
        beat_idx = 0
        start_ms = int(self._loop.time() * 1000)
        first_event = True
        try:
            async for event in self._adapter.stream(text):
                etype = event.get("type")
                if etype == "mark":
                    beat_idx += 1
                    handle.set_beat(beat_idx)
                    got_mark = True
                if first_event:
                    first_event = False
                    now_ms = int(self._loop.time() * 1000)
                    self._latency_profiler.add(now_ms - start_ms)
                # Audio chunks ignored here; mixing handled elsewhere
            # If no marks provided, synthesize them using profile
            if not got_mark and sentences:
                await self._emit_estimated_marks(sentences, handle)
        finally:
            handle.mark_finished()

    async def _emit_estimated_marks(self, sentences: list[str], handle: MarksPlaybackHandle) -> None:
        t_ms = self._profile.first_chunk_latency_ms
        await asyncio.sleep(t_ms / 1000.0)
        for i, sent in enumerate(sentences, start=1):
            words = max(1, _count_words(sent))
            dur_s = words / max(0.5, self._profile.words_per_second)
            t_ms += int(dur_s * 1000)
            handle.set_planned(i, t_ms)
            await asyncio.sleep(dur_s)
            handle.set_beat(i)
            # Include break pause after beat
            t_ms += self._profile.break_ms
            await asyncio.sleep(self._profile.break_ms / 1000.0)

    async def duck_and_play(self, text: str, *, duration_ms: int | None = None) -> None:
        # Single ducking lease: cancel any existing duck quickly with a short release
        release_ms = 50
        if self._duck_task and not self._duck_task.done():
            self._duck_task.cancel()
            try:
                await asyncio.sleep(release_ms / 1000.0)
            except asyncio.CancelledError:  # pragma: no cover - defensive
                pass

        async def _lease() -> None:
            self._duck_log.append(("start", self._loop.time()))
            try:
                await asyncio.sleep((duration_ms or 600) / 1000.0)
            finally:
                self._duck_log.append(("end", self._loop.time()))

        self._duck_task = self._loop.create_task(_lease())
        await asyncio.sleep(0)  # yield control

    async def crossfade_to_silence(self, duration_ms: int = 200) -> None:
        await asyncio.sleep(duration_ms / 1000.0)
        if self._active is not None:
            self._active.cancel()
            self._active = None
        # Also end any ducking lease to avoid overlaps
        if self._duck_task and not self._duck_task.done():
            self._duck_task.cancel()
            try:
                await asyncio.sleep(0)  # allow finally to log end
            except asyncio.CancelledError:  # pragma: no cover - defensive
                pass

    async def cancel_all(self) -> None:
        if self._active is not None:
            self._active.cancel()
            self._active = None
        if self._duck_task and not self._duck_task.done():
            self._duck_task.cancel()
            self._duck_task = None

    # ---- profiling helpers -------------------------------------------------
    def first_chunk_p95_ms(self) -> int:
        """Return the approximate P95 of first-chunk latency in ms."""
        return self._latency_profiler.p95()

    def compute_commit_guard_ms(self, *, buffer_ms: int = 75, floor_ms: int = 100) -> int:
        """Compute a dynamic commit guard band from P95 latency.

        Args:
            buffer_ms: Additional buffer on top of p95 to absorb jitter.
            floor_ms: Minimum guard band to enforce.
        Returns:
            Commit guard in milliseconds.
        """
        p95 = self.first_chunk_p95_ms()
        return max(floor_ms, p95 + max(0, buffer_ms))


class StreamingAdapter:  # pragma: no cover - protocol-like placeholder
    async def stream(self, text: str) -> AsyncIterator[dict[str, Any]]:
        """Yield streaming events. Event examples:
        {"type": "chunk", "audio": b"..."} or {"type": "mark", "name": "beat"}
        """
        yield {"type": "done"}


class _LatencyProfiler:
    """Collects first-chunk latency samples and computes an approximate P95."""

    def __init__(self, max_samples: int = 64) -> None:
        self._samples: list[int] = []
        self._max = max_samples

    def add(self, value_ms: int) -> None:
        self._samples.append(max(0, value_ms))
        if len(self._samples) > self._max:
            # Drop oldest to bound memory
            self._samples.pop(0)

    def p95(self) -> int:
        if not self._samples:
            return 0
        arr = sorted(self._samples)
        idx = int(0.95 * (len(arr) - 1))
        return arr[idx]


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _count_words(text: str) -> int:
    return len([t for t in text.strip().split() if t])
