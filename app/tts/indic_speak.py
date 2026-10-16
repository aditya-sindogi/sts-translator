"""Bodhan AI / AI4Bharat Indic-Speak (3.8B, Llama-3.2-3B + SNAC/Vocos) as a Pipecat TTS service.

Production voices include English (44, e.g. Amit, Anjali), Hindi (Amit, Kavya) and Urdu
(Saba, Zaid); the full list is in the model card. Output is 24 kHz. Not streaming: each
sentence is synthesized whole, then played.

Gated on Hugging Face (Indic Open Model License): accept the terms on the model page and
run `hf auth login` once. Hosting it as a service for others needs Bodhan AI's approval.
"""

import asyncio
import importlib.util
import sys
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

_SOURCE_RATE = 24000
_CHUNK_SAMPLES = _SOURCE_RATE // 5  # push ~200 ms frames
# Model calls run on one shared thread: one synthesis at a time on the GPU.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indic-speak")


@lru_cache
def _load(repo: str):
    """Download once and load the model's bundled TTS class (shared by all sessions)."""
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(repo)
    spec = importlib.util.spec_from_file_location("bodhan_indic_speak", f"{model_dir}/inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, model_dir)  # inference.py imports its bundled vocos/ package
    spec.loader.exec_module(module)
    logger.info(f"loaded {repo}")
    return module.TTS(model_dir)


def load(repo: str):
    return _EXECUTOR.submit(_load, repo).result()


class IndicSpeakTTSService(TTSService):
    def __init__(self, *, repo: str, voice: str, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model="indic-speak", voice=voice, language=None),
            **kwargs,
        )
        self._tts = load(repo)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        await self.start_tts_usage_metrics(text)
        loop = asyncio.get_running_loop()
        try:
            wav = await loop.run_in_executor(_EXECUTOR, lambda: self._tts(text, speaker=self._settings.voice))
        except Exception as e:
            yield ErrorFrame(error=f"Indic-Speak failed: {e}")
            return
        finally:
            await self.stop_ttfb_metrics()
        pcm = (np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype(np.int16)
        for i in range(0, len(pcm), _CHUNK_SAMPLES):
            audio = await self._resampler.resample(pcm[i : i + _CHUNK_SAMPLES].tobytes(), _SOURCE_RATE, self.sample_rate)
            yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1, context_id=context_id)
