"""NVIDIA Parakeet-TDT speech-to-text backends (wrapped by ``engines.ParakeetEngine``).

Two interchangeable backends run the same model weights:
- ``mlx``: parakeet-mlx on Apple Silicon (local development).
- ``nemo``: NVIDIA NeMo on a CUDA GPU (production, Linux). Install with ``uv sync --extra nemo``.
"""

import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from loguru import logger

from app.config import get_settings


class ParakeetBackend:
    name = "base"

    def transcribe(self, audio_f32_16k: np.ndarray) -> str:
        raise NotImplementedError


class MLXParakeetBackend(ParakeetBackend):
    name = "mlx"

    def __init__(self, model_id: str):
        # MLX streams are thread-local: the model must be loaded and run on the same
        # thread, so all MLX work goes through one dedicated worker thread (which also
        # serializes inference on the shared GPU).
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parakeet-mlx")
        self._model = self._executor.submit(self._load, model_id).result()

    @staticmethod
    def _load(model_id: str):
        from parakeet_mlx import from_pretrained

        return from_pretrained(model_id)

    def transcribe(self, audio_f32_16k: np.ndarray) -> str:
        return self._executor.submit(self._transcribe, audio_f32_16k).result()

    def _transcribe(self, audio_f32_16k: np.ndarray) -> str:
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        mel = get_logmel(mx.array(audio_f32_16k), self._model.preprocessor_config)
        return self._model.generate(mel)[0].text.strip()


class NeMoParakeetBackend(ParakeetBackend):
    name = "nemo"

    def __init__(self, model_id: str):
        import nemo.collections.asr as nemo_asr  # noqa: PLC0415  (optional dependency)

        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        self._model.eval()
        self._lock = threading.Lock()
        self._warm_up()

    def _warm_up(self):
        """Run one tiny inference now, so GPU library problems surface at startup.

        A broken cuDNN install (typically a stray LD_LIBRARY_PATH mixing CUDA versions)
        fails in the first convolution. Parakeet still runs on the GPU without cuDNN,
        a little slower, so fall back to that instead of failing every request.
        """
        import torch

        silence = np.zeros(16000, dtype=np.float32)
        try:
            self.transcribe(silence)
        except RuntimeError as e:
            if "CUDNN" not in str(e).upper():
                raise
            logger.warning(f"cuDNN failed ({e}); disabling cuDNN for Parakeet. Check LD_LIBRARY_PATH.")
            torch.backends.cudnn.enabled = False
            self.transcribe(silence)

    def transcribe(self, audio_f32_16k: np.ndarray) -> str:
        with self._lock:
            out = self._model.transcribe([audio_f32_16k], batch_size=1, verbose=False)
        first = out[0]
        return (first.text if hasattr(first, "text") else str(first)).strip()


@lru_cache
def get_backend() -> ParakeetBackend:
    """Load the model once per process; all sessions share it."""
    s = get_settings()
    backend = s.stt_backend
    if backend == "auto":
        backend = "mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "nemo"
    logger.info(f"loading Parakeet backend={backend}")
    if backend == "mlx":
        return MLXParakeetBackend(s.parakeet_mlx_model)
    if backend == "nemo":
        return NeMoParakeetBackend(s.parakeet_nemo_model)
    raise ValueError(f"unknown stt_backend {backend!r}")
