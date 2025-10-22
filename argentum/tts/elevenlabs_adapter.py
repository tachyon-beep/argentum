"""Placeholder ElevenLabs TTS adapters.

This adapter is a thin abstraction for streaming synthesis. It is deliberately
minimal at this stage and does not perform network calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ElevenLabsConfig:
    api_key: str | None = None
    voice: str | None = None
    latency_mode: str = "low"


class ElevenLabsAdapter:
    def __init__(self, config: ElevenLabsConfig | None = None) -> None:
        self.config = config or ElevenLabsConfig()

    async def synthesize(self, ssml_or_text: str, *, clip_id: str | None = None) -> bytes:  # pragma: no cover - placeholder
        _ = ssml_or_text, clip_id
        # Actual implementation to stream audio would live here.
        return b""


class ElevenLabsStreamingAdapter:
    """Streaming adapter that yields events compatible with ElevenLabsAudioController.

    Real network calls are not implemented here to keep the test suite offline.
    In production, implement `stream` to connect to the ElevenLabs streaming API
    and yield `{"type": "chunk", "audio": <bytes>}` and `{"type": "mark", ...}`
    events as they arrive.
    """

    def __init__(self, config: ElevenLabsConfig | None = None) -> None:
        self.config = config or ElevenLabsConfig()

    async def stream(self, text: str):
        """Yield streaming-like events offline.

        This offline-friendly implementation yields a first chunk event so the
        latency profiler can record a sample, then emits one `mark` per
        sentence-like unit found in the text, followed by a final `done`.

        In production, replace this method with a real ElevenLabs streaming API
        client that yields `chunk` audio bytes and `mark` events at boundaries.
        """
        # First chunk event to let controller record first-chunk latency
        yield {"type": "chunk", "audio": b""}

        # Emit a mark per sentence boundary to simulate beats
        for _ in _split_sentences(text):
            yield {"type": "mark", "name": "beat"}

        yield {"type": "done"}


def _split_sentences(text: str) -> list[str]:
    import re as _re

    parts = _re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]

