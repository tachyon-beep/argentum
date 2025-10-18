"""Audio controller interfaces and adapters.

This package provides an abstract controller for playback/crossfades/ducking
and optional provider-specific adapters (e.g., ElevenLabs).
"""

from .controller import AudioController, PlaybackHandle, NoOpAudioController

__all__ = [
    "AudioController",
    "PlaybackHandle",
    "NoOpAudioController",
]

