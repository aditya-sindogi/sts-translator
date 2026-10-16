"""Bodhan AI / AI4Bharat Indic-Transcribe (1.2B, Canary-based) as an STT engine.

Covers Indian-accented English, Hindi, Urdu, Bhojpuri and 21 more Indian languages.
Gated on Hugging Face (Indic Open Model License): accept the terms on the model page and
run `hf auth login` once. The model's own loader ships inside the download, as the model
card instructs; it runs on CUDA, Apple MPS or CPU.
"""

import asyncio
import importlib.util
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import soundfile as sf
from loguru import logger

from app.stt.engines import SAMPLE_RATE, STTEngine

# Language codes from the model card (all 25). No Awadhi: use Omnilingual ASR for that.
LANGUAGES = frozenset(
    {"as", "bhb", "bho", "bn", "brx", "doi", "en", "gu", "hi", "kn", "kok", "ks", "mai",
     "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"}
)


@lru_cache
def _load(repo: str):
    """Download once and load the model's bundled IndicTranscribe class (shared by all sessions)."""
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(repo)
    spec = importlib.util.spec_from_file_location("bodhan_indic_transcribe", f"{model_dir}/indic_transcribe.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, model_dir)  # its helper modules import each other by name
    spec.loader.exec_module(module)
    logger.info(f"loaded {repo}")
    return module.IndicTranscribe.from_pretrained(model_dir)


class IndicTranscribeEngine(STTEngine):
    name = "indic_transcribe"

    def __init__(self, repo: str, languages: list[str]):
        unknown = set(languages) - LANGUAGES
        if unknown:
            raise ValueError(f"Indic-Transcribe has no {sorted(unknown)}")
        self.languages = frozenset(languages)
        # One worker thread: model calls are serialized and never block the event loop.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indic-transcribe")
        self._asr = self._executor.submit(_load, repo).result()

    async def transcribe(self, pcm16: bytes, language: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe, pcm16, language)

    def _transcribe(self, pcm16: bytes, language: str) -> str:
        audio = np.frombuffer(pcm16, dtype=np.int16)
        # The model's public API takes a file path, so hand it a temporary 16 kHz WAV.
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, SAMPLE_RATE, subtype="PCM_16")
            out = self._asr(f.name, lang=language)
        return (out if isinstance(out, str) else str(out)).strip()
