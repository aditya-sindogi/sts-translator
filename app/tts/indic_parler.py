"""Pipecat TTS service for the team's Indic Parler-TTS server (ai4bharat/indic-parler-tts).

23 languages including Hindi, Urdu and English, ~2 voices each (keys like ``hindi_rohit``,
``urdu_divya``, ``english_mary``; see ``GET /api/voices``).

Uses ``POST /api/synthesize`` with ``format=pcm`` (raw PCM16 mono, sample rate in the
``x-sample-rate`` header). The documented batch WebSocket ``/ws/tts`` was tested and
stalled after its ``start`` message, so the HTTP endpoint is used; the body is streamed
and resampled to the pipeline's output rate as it arrives.
"""

from collections.abc import AsyncGenerator

import httpx
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

_CHUNK_BYTES = 44100 // 5 * 2  # ~200 ms at 44.1 kHz


class IndicParlerTTSService(TTSService):
    def __init__(self, *, base_url: str, voice: str, timeout_secs: float = 60.0, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model="indic-parler-tts", voice=voice, language=None),
            **kwargs,
        )
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_secs)

    def can_generate_metrics(self) -> bool:
        return True

    async def cleanup(self):
        await super().cleanup()
        await self._client.aclose()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        await self.start_tts_usage_metrics(text)
        payload = {"text": text, "description": self._settings.voice, "format": "pcm"}
        try:
            async with self._client.stream("POST", f"{self._base_url}/api/synthesize", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread())[:200]
                    yield ErrorFrame(error=f"Indic Parler TTS HTTP {resp.status_code}: {body!r}")
                    return
                source_rate = int(resp.headers.get("x-sample-rate", 44100))
                pending = b""
                async for chunk in resp.aiter_bytes(_CHUNK_BYTES):
                    await self.stop_ttfb_metrics()
                    pending += chunk
                    usable = len(pending) - len(pending) % 2  # keep whole 16-bit samples
                    if not usable:
                        continue
                    audio = await self._resampler.resample(pending[:usable], source_rate, self.sample_rate)
                    pending = pending[usable:]
                    yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1, context_id=context_id)
                logger.debug(
                    f"{self}: {resp.headers.get('x-audio-duration')}s audio, "
                    f"server time {resp.headers.get('x-processing-time')}s"
                )
        except httpx.HTTPError as e:
            yield ErrorFrame(error=f"Indic Parler TTS request failed: {e}")
        finally:
            await self.stop_ttfb_metrics()
