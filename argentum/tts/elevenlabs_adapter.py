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

    async def stream(self, text: str):  # pragma: no cover - placeholder
        _ = text
        # Yield a single done event; real implementations would stream audio & marks
        yield {"type": "done"}
