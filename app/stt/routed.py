"""A Pipecat STT service that routes each utterance to an engine by spoken language.

Routes come from ``STS_STT`` via ``app/stt/providers.py``.

The ``LanguageIDProcessor`` upstream pushes a ``LanguageDetectedFrame`` just before the
VAD stop frame, so by the time ``SegmentedSTTService`` hands us the segment we know which
language it is in. English goes to Parakeet; with the Indic ASR server enabled, Hindi and
other Indic languages go to the Zipformer server. Unknown or low-confidence segments use
the default language.

A failed transcription is retried once; a second failure marks the service unusable so the
surrounding ``ServiceSwitcher`` fails over to the CPU fallback (Moonshine).
"""

from collections.abc import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

from app.lid.speech_lid import LanguageDetectedFrame
from app.stt.engines import SAMPLE_RATE, STTEngine


class RoutedSTTService(SegmentedSTTService):
    def __init__(
        self,
        *,
        routes: dict[str, STTEngine],
        default_language: str | None = None,
        min_lid_confidence: float = 0.5,
        **kwargs,
    ):
        if not routes:
            raise ValueError("RoutedSTTService needs at least one language -> engine route")
        self._routes = dict(routes)
        self._default = default_language or ("en" if "en" in routes else next(iter(routes)))
        self._min_confidence = min_lid_confidence
        self._segment_language = self._default
        super().__init__(
            sample_rate=SAMPLE_RATE,
            settings=STTSettings(
                model="+".join(sorted({e.name for e in routes.values()})),
                language=Language(self._default),
            ),
            ttfs_p99_latency=0.3,  # Parakeet on M4 Pro ~0.07 s; re-measure per deployment
            **kwargs,
        )
        logger.info(f"STT routes: { {lang: e.name for lang, e in sorted(self._routes.items())} }")

    @property
    def wants_wav_segments(self) -> bool:
        return False  # engines take raw 16-bit PCM

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LanguageDetectedFrame):
            self._segment_language = self.language_for(frame)
        await super().process_frame(frame, direction)

    def language_for(self, frame: LanguageDetectedFrame) -> str:
        """Route to the detected language only if an engine has it and LID is confident."""
        if frame.language in self._routes and frame.confidence >= self._min_confidence:
            return frame.language
        return self._default

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Language | None = None):
        """Hook for OpenTelemetry STT spans."""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        language = self._segment_language
        self._segment_language = self._default  # each segment gets a fresh decision
        if len(audio) < SAMPLE_RATE // 5:  # < 100 ms of 16-bit audio
            yield None
            return

        engine = self._routes[language]
        await self.start_processing_metrics()
        text = None
        for attempt in (1, 2):
            try:
                text = await engine.transcribe(audio, language)
                break
            except Exception as e:
                logger.warning(f"{self}: {engine.name} attempt {attempt} failed: {e}")
                if attempt == 2:
                    await self.stop_processing_metrics()
                    # Permanent: the ServiceSwitcher fails over to the next STT service.
                    await self.push_error(
                        f"STT engine {engine.name} failed: {e}", exception=e, force_treat_as_permanent=True
                    )
                    yield None
                    return
        await self.stop_processing_metrics()

        if text:
            logger.debug(f"{self}: [{language} via {engine.name}] {text!r}")
            await self._handle_transcription(text, True, Language(language))
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language(language))
        else:
            yield None
