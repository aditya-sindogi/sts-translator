"""Builds and runs one Pipecat pipeline per WebRTC session.

    mic ─► transport in ─► VAD ─► language ID ─► STT ─► turn detection
        ─► DocumentAssistant (command → translation → summary) ─► TTS ─► transport out ─► speaker
                                    └─ text results to the browser (RTVI messages)

Which STT and TTS services run is configured in app/stt/providers.py and app/tts/providers.py.
"""

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.workers.runner import WorkerRunner

from app.config import get_settings
from app.documents.store import get_store
from app.lid.speech_lid import LanguageIDProcessor
from app.session.assistant import DocumentAssistant
from app.stt.providers import build_stt_service
from app.tts.providers import build_tts_service


async def run_bot(connection: SmallWebRTCConnection, session_id: str):
    settings = get_settings()
    store = get_store()

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    tts, tts_for_language = build_tts_service()
    assistant = DocumentAssistant(
        session_id=session_id,
        get_document=store.latest_for_session,
        save_result=store.put_result,
        tts_for_language=tts_for_language,
    )

    pipeline = Pipeline(
        [
            transport.input(),  # 1. audio from the browser mic
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),  # 2. when is the user speaking?
            LanguageIDProcessor(min_speech_secs=settings.lid_min_speech_secs),  # 3. which language?
            build_stt_service(),  # 4. speech -> text
            UserTurnProcessor(),  # 5. has the user finished their turn?
            assistant,  # 6. command -> translation -> summary (text out)
            tts,  # 7. text -> speech
            transport.output(),  # 8. audio to the browser speaker
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
