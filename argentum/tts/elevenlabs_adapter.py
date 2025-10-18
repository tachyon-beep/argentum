"""Placeholder ElevenLabs TTS adapter.

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

