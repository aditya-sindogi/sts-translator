"""Client for the team's k2-sherpa streaming Zipformer ASR server (12 Indic languages + English).

Protocol (asr/ASR.md on the GPU box): on connect the server sends ``ready``; the client
sends ``{"type":"start","lang":...}``, then raw PCM16LE 16 kHz frames (250 ms = 8000 bytes,
one ``partial`` reply each), then ``{"type":"end"}`` and receives ``final``.

Because language ID runs on the whole VAD segment before transcription starts, we already
know the language when the segment arrives, so each utterance is streamed to the server
as fast as the network allows rather than in real time.
"""

import asyncio
import json
import time

import websockets
from loguru import logger

from app.stt.engines import STTEngine

# ISO 639-1 -> server language names (from the server's "ready" message).
ISO_TO_SERVER = {
    "as": "assamese",
    "bn": "bengali",
    "en": "english",
    "gu": "gujarati",
    "hi": "hindi",
    "kn": "kannada",
    "ml": "malayalam",
    "mr": "marathi",
    "or": "odia",
    "pa": "punjabi",
    "ta": "tamil",
    "te": "telugu",
}
_CHUNK_BYTES = 8000  # 250 ms
# The causal Zipformer emits the last tokens only after it sees more audio; without tail
# padding the final word is dropped (measured: "...in the" for "...into English").
_TAIL_PADDING = bytes(int(16000 * 0.8) * 2)


class IndicZipformerEngine(STTEngine):
    name = "indic-zipformer"

    def __init__(self, url: str, languages: list[str], timeout_secs: float = 10.0):
        unknown = set(languages) - ISO_TO_SERVER.keys()
        if unknown:
            raise ValueError(f"Indic Zipformer has no model for {sorted(unknown)}")
        self.languages = frozenset(languages)
        self._url = url
        self._timeout = timeout_secs

    async def transcribe(self, pcm16: bytes, language: str) -> str:
        lang = ISO_TO_SERVER[language]
        started = time.perf_counter()
        async with websockets.connect(self._url, open_timeout=self._timeout, max_size=2**22) as ws:
            ready = json.loads(await _recv(ws, self._timeout))
            if ready.get("type") != "ready" or lang not in ready.get("langs", []):
                raise RuntimeError(f"ASR server not ready for {lang}: {ready}")
            await ws.send(json.dumps({"type": "start", "lang": lang}))
            ack = json.loads(await _recv(ws, self._timeout))
            if ack.get("type") != "start_ack":
                raise RuntimeError(f"unexpected ASR reply: {ack}")

            audio = pcm16 + _TAIL_PADDING
            for i in range(0, len(audio), _CHUNK_BYTES):
                await ws.send(audio[i : i + _CHUNK_BYTES])
            await ws.send(json.dumps({"type": "end"}))

            while True:  # drain partials until the final hypothesis
                msg = json.loads(await _recv(ws, self._timeout))
                if msg.get("type") == "final":
                    text = (msg.get("text") or "").strip()
                    break
                if msg.get("type") == "error":
                    raise RuntimeError(f"ASR server error: {msg.get('message')}")
        logger.debug(f"zipformer[{lang}] {len(pcm16) / 32000:.1f}s audio in {time.perf_counter() - started:.2f}s")
        return text


async def _recv(ws, timeout: float):
    return await asyncio.wait_for(ws.recv(), timeout)
