"""Tests for audio controller factory selection."""

from argentum.audio.factory import get_audio_controller
from argentum.audio.controller import SimAudioController
from argentum.audio.elevenlabs_controller import ElevenLabsAudioController


def test_factory_default_sim_controller():
    ctrl = get_audio_controller({})
    assert isinstance(ctrl, SimAudioController)


def test_factory_elevenlabs_controller():
    cfg = {"conversation": {"tts": {"provider": "elevenlabs", "voice": "default", "latency_mode": "low"}}}
    ctrl = get_audio_controller(cfg)
    assert isinstance(ctrl, ElevenLabsAudioController)

