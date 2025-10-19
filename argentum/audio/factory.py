"""Factory helpers to construct an AudioController from config.

By default, returns a SimAudioController. If a provider of "elevenlabs" is
specified in the config, returns an ElevenLabsAudioController with a placeholder
streaming adapter suitable for offline tests. Production code can pass a real
adapter.
"""

from __future__ import annotations

from typing import Any

from .controller import AudioController
from .controller import SimAudioController
from .elevenlabs_controller import ElevenLabsAudioController, ElevenLabsVoiceProfile
from argentum.tts.elevenlabs_adapter import ElevenLabsConfig, ElevenLabsStreamingAdapter


def get_audio_controller(config: dict[str, Any] | None = None) -> AudioController:
    cfg = config or {}
    convo = cfg.get("conversation") or {}
    tts = convo.get("tts") or {}
    provider = (tts.get("provider") or "").lower()
    if provider == "elevenlabs":
        voice = tts.get("voice") or None
        latency_mode = tts.get("latency_mode") or "low"
        profile = ElevenLabsVoiceProfile()
        adapter = ElevenLabsStreamingAdapter(ElevenLabsConfig(api_key=None, voice=voice, latency_mode=latency_mode))
        return ElevenLabsAudioController(adapter, profile=profile)
    # default: simulated controller
    return SimAudioController()

