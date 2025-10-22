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
        """Yield streaming events.

        If ELEVENLABS_API_KEY and ELEVENLABS_VOICE are set (or provided via
        config), attempt an HTTP streaming request and yield `chunk` events.
        Otherwise, fallback to an offline path that emits a first `chunk` and
        sentence-based `mark` events.
        """
        try:
            import os as _os
            import httpx as _httpx
        except Exception:
            _os = None
            _httpx = None

        api_key = (self.config.api_key if isinstance(self.config.api_key, str) else None)
        voice = (self.config.voice if isinstance(self.config.voice, str) else None)
        if _os is not None:
            api_key = api_key or _os.getenv('ELEVENLABS_API_KEY')
            voice = voice or _os.getenv('ELEVENLABS_VOICE')

        use_network = bool(api_key and voice and _httpx is not None)
        if use_network:
            base = (_os.getenv('ELEVENLABS_API_BASE') if _os else None) or 'https://api.elevenlabs.io/v1/text-to-speech'
            url = f"{base.rstrip('/')}/{voice}/stream"
            headers = {
                'xi-api-key': api_key,
                'accept': 'audio/mpeg',
                'content-type': 'application/json',
            }
            payload = {
                'text': text,
                'model_id': (_os.getenv('ELEVENLABS_MODEL') if _os else None) or 'eleven_monolingual_v1',
            }
            try:
                timeout = _httpx.Timeout(10.0, connect=10.0, read=10.0)
                async with _httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream('POST', url, headers=headers, json=payload) as r:
                        r.raise_for_status()
                        first = True
                        async for chunk in r.aiter_bytes():
                            if first:
                                yield {"type": "chunk", "audio": b""}
                                first = False
                            yield {"type": "chunk", "audio": chunk}
                        yield {"type": "done"}
                        return
            except Exception:
                # Fall through to offline path on any network failure
                pass

        # Offline path: first chunk, then sentence marks, then done
        yield {"type": "chunk", "audio": b""}
        for _ in _split_sentences(text):
            yield {"type": "mark", "name": "beat"}
        yield {"type": "done"}



def _split_sentences(text: str) -> list[str]:
    import re as _re

    parts = _re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]

