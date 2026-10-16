"""Speech-to-text engines: plain async "PCM segment in, text out" adapters.

Engines know nothing about Pipecat. ``RoutedSTTService`` wraps them as one Pipecat
service and picks an engine per utterance from the detected spoken language.
Register new engines in ``app/stt/providers.py``.
"""

import asyncio

import numpy as np

SAMPLE_RATE = 16000


class STTEngine:
    """Transcribes one complete utterance of 16 kHz mono PCM16 audio."""

    name = "base"
    languages: frozenset[str] = frozenset()  # ISO 639-1 codes this engine can transcribe

    async def transcribe(self, pcm16: bytes, language: str) -> str:
        raise NotImplementedError


class ParakeetEngine(STTEngine):
    """NVIDIA Parakeet-TDT (English), running in-process on MLX or NeMo."""

    name = "parakeet"
    languages = frozenset({"en"})

    def __init__(self, backend=None):
        from app.stt.parakeet import get_backend

        self._backend = backend or get_backend()
        self.name = f"parakeet-{self._backend.name}"

    async def transcribe(self, pcm16: bytes, language: str) -> str:
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return await asyncio.to_thread(self._backend.transcribe, samples)
