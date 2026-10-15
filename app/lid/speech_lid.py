"""Spoken language identification on VAD-bounded speech segments.

The LID model (SpeechBrain VoxLingua107 ECAPA, Apache-2.0) runs on CPU in ~15 ms per
segment. ``LanguageIDProcessor`` sits right after the VAD in the Pipecat pipeline: it
buffers audio while the user speaks, and when the VAD reports the end of speech it
classifies the segment and pushes a ``LanguageDetectedFrame`` *before* forwarding the
VAD stop frame, so downstream processors always see the language before the transcript.
"""

import asyncio
import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    SystemFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.config import get_settings

_LID_SAMPLE_RATE = 16000
_PREROLL_SECS = 0.5  # VAD fires after speech has started; keep a little audio before it


@dataclass
class LanguageDetectedFrame(SystemFrame):
    """Result of spoken language ID for one speech segment.

    Parameters:
        language: ISO 639-1 code of the top prediction (None if the segment was too short).
        confidence: Posterior probability of the top prediction.
        duration_secs: Length of the speech segment.
        top: The three best (language, probability) pairs, for logging/UI.
    """

    language: str | None
    confidence: float
    duration_secs: float
    top: list[tuple[str, float]]


class SpokenLanguageIdentifier:
    """Thread-safe wrapper around the SpeechBrain classifier (loaded once per process)."""

    def __init__(self, source: str, savedir: str):
        from speechbrain.inference.classifiers import EncoderClassifier

        self._model = EncoderClassifier.from_hparams(source=source, savedir=savedir, run_opts={"device": "cpu"})
        self._labels = self._model.hparams.label_encoder.ind2lab
        self._lock = threading.Lock()

    def classify(self, audio_f32_16k: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        import torch

        with self._lock, torch.inference_mode():
            log_probs = self._model.classify_batch(torch.from_numpy(audio_f32_16k).unsqueeze(0))[0][0]
        probs = torch.softmax(log_probs, dim=-1)
        values, indices = torch.topk(probs, 3)
        top = [(self._labels[int(i)].split(":")[0], float(v)) for v, i in zip(values, indices)]
        return top[0][0], top[0][1], top


@lru_cache
def get_identifier() -> SpokenLanguageIdentifier:
    s = get_settings()
    return SpokenLanguageIdentifier(s.lid_model, str(s.models_dir / "lid-voxlingua107"))


class LanguageIDProcessor(FrameProcessor):
    def __init__(self, *, min_speech_secs: float, **kwargs):
        super().__init__(**kwargs)
        self._min_speech_secs = min_speech_secs
        self._identifier = get_identifier()
        self._buffer = bytearray()
        self._sample_rate = _LID_SAMPLE_RATE
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._sample_rate = frame.sample_rate
            self._buffer += frame.audio
            if not self._speaking:
                keep = int(self._sample_rate * _PREROLL_SECS) * 2
                if len(self._buffer) > keep:
                    del self._buffer[: len(self._buffer) - keep]
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
            pcm = bytes(self._buffer)
            self._buffer.clear()
            await self.push_frame(await self._identify(pcm))

        await self.push_frame(frame, direction)

    async def _identify(self, pcm: bytes) -> LanguageDetectedFrame:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self._sample_rate != _LID_SAMPLE_RATE and len(audio):
            n = int(len(audio) * _LID_SAMPLE_RATE / self._sample_rate)
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        duration = len(audio) / _LID_SAMPLE_RATE
        if duration < self._min_speech_secs:
            return LanguageDetectedFrame(language=None, confidence=0.0, duration_secs=duration, top=[])
        try:
            lang, conf, top = await asyncio.to_thread(self._identifier.classify, audio)
        except Exception as e:  # LID must never break the voice loop
            logger.error(f"{self}: language ID failed: {e}")
            return LanguageDetectedFrame(language=None, confidence=0.0, duration_secs=duration, top=[])
        logger.debug(f"{self}: language {lang} ({conf:.2f}) over {duration:.1f}s, top={top}")
        return LanguageDetectedFrame(language=lang, confidence=conf, duration_secs=duration, top=top)
