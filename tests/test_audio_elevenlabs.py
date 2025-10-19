"""Tests for the ElevenLabsAudioController using a fake streaming adapter."""

import asyncio

import pytest

from argentum.audio.elevenlabs_controller import (
    ElevenLabsAudioController,
    ElevenLabsVoiceProfile,
    StreamingAdapter,
)


class FakeAdapter(StreamingAdapter):
    def __init__(self, mark_times_ms: list[int]) -> None:
        self._mark_times_ms = mark_times_ms

    async def stream(self, text: str):  # type: ignore[override]
        # Emit marks at specified times; no audio chunks
        for t_ms in self._mark_times_ms:
            await asyncio.sleep(t_ms / 1000.0)
            yield {"type": "mark", "time_ms": t_ms}
        yield {"type": "done"}


@pytest.mark.asyncio
async def test_elevenlabs_controller_emits_marks():
    adapter = FakeAdapter([100, 300])
    ctrl = ElevenLabsAudioController(adapter, profile=ElevenLabsVoiceProfile())
    handle = await ctrl.play("Hello. World.")
    # Wait for first and second beat marks
    await handle.wait_for_beat(1)
    await handle.wait_for_beat(2)
    await handle.finish()


@pytest.mark.asyncio
async def test_ducking_single_lease_no_overlap():
    adapter = FakeAdapter([])
    ctrl = ElevenLabsAudioController(adapter, profile=ElevenLabsVoiceProfile())
    # Kick off a dummy playback
    _ = await ctrl.play("Hello world.")
    # Issue two ducks rapidly
    await ctrl.duck_and_play("hi", duration_ms=200)
    await ctrl.duck_and_play("again", duration_ms=200)
    # Wait a bit to allow both to complete
    await asyncio.sleep(0.6)
    # Extract log and ensure end of first <= start of second (no overlap)
    starts = [t for label, t in ctrl._duck_log if label == "start"]
    ends = [t for label, t in ctrl._duck_log if label == "end"]
    assert len(starts) >= 2 and len(ends) >= 2
    assert ends[0] <= starts[1] + 0.001  # allow tiny scheduling jitter


@pytest.mark.asyncio
async def test_crossfade_cancels_duck_cleanly():
    adapter = FakeAdapter([])
    ctrl = ElevenLabsAudioController(adapter, profile=ElevenLabsVoiceProfile())
    _ = await ctrl.play("Hello world.")
    await ctrl.duck_and_play("hi", duration_ms=500)
    # Immediately crossfade to silence which should cancel the ducking lease
    await ctrl.crossfade_to_silence(duration_ms=0)
    await asyncio.sleep(0.1)
    starts = [t for label, t in ctrl._duck_log if label == "start"]
    ends = [t for label, t in ctrl._duck_log if label == "end"]
    assert len(ends) >= len(starts)  # end should be logged even if cancelled


@pytest.mark.asyncio
async def test_estimated_marks_with_drift_recording():
    # Adapter with no marks forces estimated-beat path
    adapter = FakeAdapter([])
    profile = ElevenLabsVoiceProfile(words_per_second=3.0, break_ms=50, first_chunk_latency_ms=50)
    ctrl = ElevenLabsAudioController(adapter, profile=profile)
    handle = await ctrl.play("One. Two.")
    await handle.wait_for_beat(1)
    await handle.wait_for_beat(2)
    await handle.finish()
    drift = handle.drift_ms()
    # Planned timestamps should be set and drift reasonably bounded
    assert len(drift) >= 2
    assert all(isinstance(x, int) for x in drift)
    # Drift should be within +/- 200ms for this simple simulation
    assert all(abs(x) < 200 for x in drift if x)


def test_latency_profiler_p95_exposure():
    # Access the profiling helper via controller
    adapter = FakeAdapter([])
    ctrl = ElevenLabsAudioController(adapter, profile=ElevenLabsVoiceProfile())
    # Manually add samples through private API to simulate distribution
    ctrl._latency_profiler.add(50)   # type: ignore[attr-defined]
    ctrl._latency_profiler.add(60)   # type: ignore[attr-defined]
    ctrl._latency_profiler.add(70)   # type: ignore[attr-defined]
    ctrl._latency_profiler.add(200)  # type: ignore[attr-defined]
    p95 = ctrl.first_chunk_p95_ms()
    assert isinstance(p95, int)
    assert p95 >= 70
    guard = ctrl.compute_commit_guard_ms(buffer_ms=50, floor_ms=80)
    assert guard >= 120  # p95 + buffer


@pytest.mark.asyncio
async def test_last_millisecond_cancel_sequence():
    adapter = FakeAdapter([])
    ctrl = ElevenLabsAudioController(adapter, profile=ElevenLabsVoiceProfile())
    _ = await ctrl.play("Sentence one. Sentence two.")
    # Start a long duck, then quickly crossfade to cancel it near-immediately
    await ctrl.duck_and_play("interrupting", duration_ms=500)
    await ctrl.crossfade_to_silence(duration_ms=10)
    await asyncio.sleep(0.1)
    # There should be at least one start/end pair logged; ends shouldn't be fewer than starts
    starts = [t for label, t in ctrl._duck_log if label == "start"]
    ends = [t for label, t in ctrl._duck_log if label == "end"]
    assert len(ends) >= len(starts)
