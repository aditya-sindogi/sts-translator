"""Bodhan AI / AI4Bharat Indic-Speak (3.8B, Llama-3.2-3B + SNAC/Vocos) as a streaming Pipecat TTS service.

Production voices include English (44, e.g. Amit, Anjali), Hindi (Amit, Kavya) and Urdu
(Saba, Zaid); the full list is in the model card. Output is 24 kHz.

The model writes speech as audio tokens: every 7 tokens are one 85 ms SNAC frame, which a
small Vocos decoder turns into 2048 samples. The bundled ``inference.py`` waits for the whole
sentence (~2x slower than real time with plain ``generate``). Here we:
  - generate with one fixed-size static KV cache, which Transformers compiles once
    (measured on an L40S: 42 -> 71 tokens/s, i.e. close to the 82 tokens/s of real time);
    a fixed size matters: a cache sized per prompt recompiles for every new length;
  - decode frames while tokens are still being generated, with a little left context and
    look-ahead so chunk boundaries are seamless, and stream each chunk to the speaker;
  - buffer just enough audio before the first chunk that playback never catches up with
    generation, sized from the sentence's expected length and the measured token rate.

Gated on Hugging Face (Indic Open Model License): accept the terms on the model page and
run `hf auth login` once. Hosting it as a service for others needs Bodhan AI's approval.
"""

import asyncio
import importlib.util
import re
import sys
import time
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import torch
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

SAMPLE_RATE = 24000
FRAME_TOKENS = 7  # SNAC tokens per frame
FRAME_SAMPLES = 2048  # samples per frame (4 latent steps x 512)
MIN_FIRST_CHUNK_FRAMES = 4  # always buffer at least ~0.35 s
CHUNK_FRAMES = 3  # then stream every ~0.25 s
REALTIME_TOKENS_PER_SEC = FRAME_TOKENS * SAMPLE_RATE / FRAME_SAMPLES  # ~82
FRAMES_PER_CHAR = 1.0  # measured ~0.8-1.0 audio frames per character (en/hi/ur)
MAX_NEW_TOKENS = 2520
CACHE_LEN = 3072  # prompt + MAX_NEW_TOKENS fits; one size => one compile
# Long sentences are spoken clause by clause: the first clause starts playing sooner, and
# any catch-up time lands at a comma, where a short pause sounds natural.
CLAUSE_SPLIT_CHARS = 60
MIN_CLAUSE_CHARS = 25
_CLAUSE_END_RE = re.compile(r"(?<=[,;:\u060c])\s+")  # , ; : and the Urdu comma ،
LOOKAHEAD_FRAMES = 2  # decode a little past what we emit, so the chunk edge is clean
CONTEXT_FRAMES = 8  # and a little before it, for the decoder's left context

# All model work runs on one thread: one synthesis at a time on the GPU.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indic-speak")


class IndicSpeakStreamer:
    """Loads Indic-Speak once and synthesizes a sentence as a sequence of PCM chunks."""

    def __init__(self, repo: str):
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(repo)
        spec = importlib.util.spec_from_file_location("bodhan_indic_speak", f"{model_dir}/inference.py")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, model_dir)  # inference.py imports its bundled vocos/ package
        spec.loader.exec_module(module)
        self._build_prompt = module.build_prompt
        self._ids_to_codes = module.ids_to_codes
        self._tts = module.TTS(model_dir)
        self._base = self._tts._base
        self._audio_end = self._base + FRAME_TOKENS * 4096
        self._static = self._tts.device == "cuda"  # static cache + compile only pays off on CUDA
        self._cache = None
        if self._static:
            from transformers import StaticCache

            self._cache = StaticCache(config=self._tts.lm.config, max_cache_len=CACHE_LEN)
        self.tokens_per_sec = 60.0  # running estimate, refined after every sentence
        logger.info(f"loaded {repo} on {self._tts.device} (static cache: {self._static})")
        self._warm_up()

    def _warm_up(self):
        """Trigger the one-time compile now rather than on the first user's sentence."""
        started = time.perf_counter()
        for text in ("Hello.", "This is a warm-up sentence for the speech model.", "नमस्ते, यह एक छोटा सा परीक्षण वाक्य है।"):
            self.synthesize(text, "Amit", lambda pcm: None)
        logger.info(f"Indic-Speak warm-up done in {time.perf_counter() - started:.1f}s")

    def synthesize(self, text: str, speaker: str, emit: Callable[[bytes], None]) -> None:
        """Generate speech for ``text``, calling ``emit`` with 16-bit PCM chunks as they're ready."""
        tok = self._tts.tok
        prompt = self._build_prompt(tok, text, speaker)
        ids = torch.tensor([prompt], device=self._tts.device)
        streamer = _FrameStreamer(self, emit, self._first_chunk_frames(text))
        kwargs = {}
        if self._cache is not None:
            self._cache.reset()
            kwargs["past_key_values"] = self._cache
        started = time.perf_counter()
        with torch.no_grad():
            self._tts.lm.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=self._token_budget(text),
                eos_token_id=[self._tts._eos, tok.eos_token_id],
                pad_token_id=tok.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                streamer=streamer,
                **kwargs,
            )
        streamer.flush()
        elapsed = time.perf_counter() - started
        if streamer.token_count > 50 and elapsed > 0:
            rate = streamer.token_count / elapsed
            self.tokens_per_sec = 0.7 * self.tokens_per_sec + 0.3 * rate

    @staticmethod
    def _token_budget(text: str) -> int:
        """Cap generation at ~2.5x the expected audio: an occasional runaway (the model
        babbling past the end of the text) is cut off in seconds instead of ~30 s."""
        expected_tokens = len(text) * FRAMES_PER_CHAR * FRAME_TOKENS
        return int(min(MAX_NEW_TOKENS, max(140, expected_tokens * 2.5)))

    def _first_chunk_frames(self, text: str) -> int:
        """Frames to buffer so that, at the current token rate, playback never runs dry."""
        expected = len(text) * FRAMES_PER_CHAR
        slowdown = max(0.0, 1 - self.tokens_per_sec / REALTIME_TOKENS_PER_SEC)  # fraction behind real time
        return max(MIN_FIRST_CHUNK_FRAMES, int(expected * slowdown) + 2)

    def decode(self, tokens: list[int], first: int, last: int) -> np.ndarray:
        """Decode frames [first, last) to float32 samples."""
        codes = self._ids_to_codes(tokens[first * FRAME_TOKENS : last * FRAME_TOKENS], self._base, self._tts.device)
        with torch.no_grad():
            z_q = self._tts.snac.quantizer.from_codes(codes)
            wav = self._tts.vocos(z_q.float())
        return wav[0, 0].clamp(-1, 1).float().cpu().numpy()

    def is_audio(self, token: int) -> bool:
        return self._base <= token < self._audio_end


class _FrameStreamer:
    """Transformers streamer: collects audio tokens and emits decoded chunks as frames complete."""

    def __init__(self, model: IndicSpeakStreamer, emit: Callable[[bytes], None], first_chunk_frames: int):
        self._model = model
        self._emit = emit
        self._first_chunk_frames = first_chunk_frames
        self._tokens: list[int] = []
        self._emitted = 0  # frames already sent
        self._seen_prompt = False
        self._done = False

    def put(self, value: torch.Tensor):
        if not self._seen_prompt:  # generate() passes the prompt first
            self._seen_prompt = True
            return
        if self._done:
            return
        for token in value.reshape(-1).tolist():
            if not self._model.is_audio(token):
                self._done = True  # end of speech
                break
            self._tokens.append(token)
        frames = len(self._tokens) // FRAME_TOKENS
        ready = frames - LOOKAHEAD_FRAMES  # frames we can emit with look-ahead available
        need = self._first_chunk_frames if self._emitted == 0 else CHUNK_FRAMES
        if ready - self._emitted >= need:
            self._send(self._emitted, ready, frames)

    def end(self):
        pass  # flush() is called explicitly after generate() returns

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    def flush(self):
        frames = len(self._tokens) // FRAME_TOKENS
        if frames > self._emitted:
            self._send(self._emitted, frames, frames)

    def _send(self, start: int, stop: int, available: int):
        first = max(0, start - CONTEXT_FRAMES)
        wav = self._model.decode(self._tokens, first, available)
        chunk = wav[(start - first) * FRAME_SAMPLES : (stop - first) * FRAME_SAMPLES]
        self._emitted = stop
        if len(chunk):
            self._emit((chunk * 32767).astype(np.int16).tobytes())


@lru_cache
def _load(repo: str) -> IndicSpeakStreamer:
    return IndicSpeakStreamer(repo)


def load(repo: str) -> IndicSpeakStreamer:
    """Load (once per process) on the model thread."""
    return _EXECUTOR.submit(_load, repo).result()


class IndicSpeakTTSService(TTSService):
    def __init__(self, *, repo: str, voice: str, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model="indic-speak", voice=voice, language=None),
            **kwargs,
        )
        self._model = load(repo)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        await self.start_tts_usage_metrics(text)
        loop = asyncio.get_running_loop()
        chunks: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()

        def emit(pcm: bytes):
            loop.call_soon_threadsafe(chunks.put_nowait, pcm)

        def work():
            try:
                for clause in split_clauses(text):
                    self._model.synthesize(clause, self._settings.voice, emit)
                loop.call_soon_threadsafe(chunks.put_nowait, None)
            except Exception as e:  # surfaced as an ErrorFrame below
                loop.call_soon_threadsafe(chunks.put_nowait, e)

        loop.run_in_executor(_EXECUTOR, work)
        first = True
        while (item := await chunks.get()) is not None:
            if isinstance(item, Exception):
                yield ErrorFrame(error=f"Indic-Speak failed: {item}")
                break
            if first:
                await self.stop_ttfb_metrics()
                first = False
            audio = await self._resampler.resample(item, SAMPLE_RATE, self.sample_rate)
            yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1, context_id=context_id)
        if first:
            await self.stop_ttfb_metrics()


def split_clauses(text: str) -> list[str]:
    """Split a long sentence at commas/semicolons into clauses of a useful size."""
    if len(text) <= CLAUSE_SPLIT_CHARS:
        return [text]
    clauses: list[str] = []
    for part in _CLAUSE_END_RE.split(text):
        # Tiny fragments sound odd on their own and can make the model babble: join them.
        if clauses and len(clauses[-1]) < MIN_CLAUSE_CHARS:
            clauses[-1] = f"{clauses[-1]} {part}"
        else:
            clauses.append(part)
    if len(clauses) > 1 and len(clauses[-1]) < MIN_CLAUSE_CHARS:
        clauses[-2:] = [f"{clauses[-2]} {clauses[-1]}"]
    return [c for c in clauses if c.strip()]
