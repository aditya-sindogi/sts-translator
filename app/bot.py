"""Builds and runs one Pipecat pipeline per WebRTC session.

    mic ─► SmallWebRTC in ─► Silero VAD ─► Language ID
        ─► STT: routed by spoken language (en → Parakeet, hi → Indic Zipformer*), Moonshine failover
        ─► Smart-turn ─► DocumentAssistant (command → translation → summary)
        ─► TTS: switched by text language (Kokoro, Indic Parler-TTS*) ─► SmallWebRTC out ─► speaker
                         └─ RTVI server messages (text output) over the data channel
    * optional remote GPU services, enabled by STS_INDIC_ASR_URL / STS_INDIC_TTS_URL
"""

from collections.abc import Callable

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import (
    ServiceSwitcher,
    ServiceSwitcherStrategyFailover,
    ServiceSwitcherStrategyManual,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.moonshine.stt import MoonshineSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.workers.runner import WorkerRunner

from app.config import get_settings
from app.documents.store import get_store
from app.lid.speech_lid import LanguageIDProcessor
from app.session.assistant import DocumentAssistant
from app.stt.engines import ParakeetEngine
from app.stt.indic_zipformer import IndicZipformerEngine
from app.stt.routed import RoutedSTTService
from app.tts.indic_parler import IndicParlerTTSService

TTSSelector = Callable[[str], FrameProcessor | None]


def build_stt():
    settings = get_settings()
    engines = [ParakeetEngine()]
    if settings.indic_asr_url:
        engines.append(IndicZipformerEngine(settings.indic_asr_url, settings.indic_asr_languages))
    services = [RoutedSTTService(engines=engines, default_language="en")]
    if settings.stt_moonshine_fallback:
        services.append(MoonshineSTTService())
    if len(services) == 1:
        return services[0]
    switcher = ServiceSwitcher(services=services, strategy_type=ServiceSwitcherStrategyFailover)

    @switcher.strategy.event_handler("on_service_switched")
    async def on_switched(strategy, service):
        logger.warning(f"STT failover: now using {service}")

    return switcher


def build_tts() -> tuple[FrameProcessor, TTSSelector]:
    """Return the TTS processor and a selector mapping a text language to its service.

    With more than one service they sit in a manual ServiceSwitcher; the assistant pushes
    a ``ManuallySwitchServiceFrame`` before speaking text in a different language.
    """
    settings = get_settings()
    by_language: dict[str, FrameProcessor] = {}
    if settings.tts_english_provider == "kokoro" or not settings.indic_tts_url:
        by_language["en"] = KokoroTTSService(
            settings=KokoroTTSService.Settings(voice=settings.kokoro_voice),
            stop_frame_timeout_s=8.0,  # CPU synthesis of a long sentence can exceed the 3 s default
        )
    if settings.indic_tts_url:
        for lang, voice in settings.indic_tts_voices.items():
            if lang not in by_language:
                by_language[lang] = IndicParlerTTSService(
                    base_url=settings.indic_tts_url,
                    voice=voice,
                    stop_frame_timeout_s=15.0,  # remote GPU synthesis runs ~1.5x real time
                )
    services = list(by_language.values())
    if len(services) == 1:
        return services[0], lambda lang: None
    switcher = ServiceSwitcher(services=services, strategy_type=ServiceSwitcherStrategyManual)
    logger.info(f"TTS routes: { {lang: type(svc).__name__ for lang, svc in by_language.items()} }")
    return switcher, lambda lang: by_language.get(lang, by_language.get("en"))


async def run_bot(connection: SmallWebRTCConnection, session_id: str):
    settings = get_settings()
    store = get_store()

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    tts, tts_for_language = build_tts()
    assistant = DocumentAssistant(
        session_id=session_id,
        get_document=store.latest_for_session,
        save_result=store.put_result,
        tts_for_language=tts_for_language,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            LanguageIDProcessor(min_speech_secs=settings.lid_min_speech_secs),
            build_stt(),
            UserTurnProcessor(),
            assistant,
            tts,
            transport.output(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=None,  # translation jobs can be long and silent
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"session {session_id}: client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"session {session_id}: client disconnected")
        await runner.cancel()

    await runner.run()
