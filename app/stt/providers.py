"""Speech-to-text providers: the one place to add or swap an STT service.

The pipeline gets one Pipecat STT service from ``build_stt_service()``. Inside it, each
utterance goes to the engine configured for its detected spoken language (``STS_STT``):

    STS_STT='{"en": "parakeet"}'                              # default
    STS_STT='{"en": "parakeet", "hi": "indic_zipformer"}'     # + Hindi via the team server
    STS_STT='{"en": "indic_transcribe", "hi": "indic_transcribe", "ur": "indic_transcribe"}'

To add a provider (e.g. Wispr, Deepgram, Whisper):
  1. Write an ``STTEngine`` subclass (see engines.py): ``async transcribe(pcm16, language) -> str``.
     It receives one complete utterance of 16 kHz mono 16-bit audio.
  2. Register a factory for it in ``ENGINES`` below.
  3. Point a language at it in ``STS_STT``.
"""

from collections.abc import Callable

from loguru import logger

from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyFailover
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.moonshine.stt import MoonshineSTTService

from app.config import get_settings
from app.stt.engines import ParakeetEngine, STTEngine
from app.stt.indic_transcribe import IndicTranscribeEngine
from app.stt.indic_zipformer import IndicZipformerEngine
from app.stt.parakeet import try_get_backend
from app.stt.routed import RoutedSTTService


def _parakeet(languages: list[str]) -> STTEngine:
    backend = try_get_backend()
    if backend is None:
        raise RuntimeError("Parakeet could not load on this machine")
    return ParakeetEngine(backend)


def _indic_zipformer(languages: list[str]) -> STTEngine:
    return IndicZipformerEngine(get_settings().indic_asr_url, languages)


def _indic_transcribe(languages: list[str]) -> STTEngine:
    return IndicTranscribeEngine(get_settings().indic_transcribe_model, languages)


# Engine name -> factory(languages it will serve) -> STTEngine
ENGINES: dict[str, Callable[[list[str]], STTEngine]] = {
    "parakeet": _parakeet,
    "indic_zipformer": _indic_zipformer,
    "indic_transcribe": _indic_transcribe,
}

# Pipecat built-in services used as a whole-service fallback if the main STT fails.
FALLBACKS: dict[str, Callable[[], FrameProcessor]] = {
    "moonshine": MoonshineSTTService,
}


def build_engines() -> dict[str, STTEngine]:
    """Language -> engine, from STS_STT. Engines that can't load are skipped with a warning."""
    by_engine: dict[str, list[str]] = {}
    for lang, name in get_settings().stt.items():
        if name not in ENGINES:
            raise ValueError(f"STS_STT: unknown STT engine {name!r} for {lang!r}; known: {sorted(ENGINES)}")
        by_engine.setdefault(name, []).append(lang)

    routes: dict[str, STTEngine] = {}
    for name, languages in by_engine.items():
        try:
            engine = ENGINES[name](languages)
        except Exception as e:
            logger.warning(f"STT engine {name!r} unavailable ({e}); languages {languages} lose their engine")
            continue
        unsupported = set(languages) - engine.languages
        if unsupported:
            raise ValueError(f"STS_STT: {name!r} can't transcribe {sorted(unsupported)}")
        for lang in languages:
            routes[lang] = engine
    return routes


def build_stt_service() -> FrameProcessor:
    """The STT stage of the pipeline: routed engines, with an optional fallback service."""
    routes = build_engines()
    fallback_name = get_settings().stt_fallback
    fallback = FALLBACKS[fallback_name]() if fallback_name else None

    if not routes:
        if fallback is None:
            raise RuntimeError("No STT engine could load and no fallback is configured")
        logger.warning(f"Using fallback STT only: {fallback_name}")
        return fallback

    primary = RoutedSTTService(routes=routes)
    if fallback is None:
        return primary
    switcher = ServiceSwitcher(services=[primary, fallback], strategy_type=ServiceSwitcherStrategyFailover)

    @switcher.strategy.event_handler("on_service_switched")
    async def on_switched(strategy, service):
        logger.warning(f"STT failover: now using {service}")

    return switcher


def warm_up() -> None:
    """Load engines and download the fallback model before the first caller arrives."""
    build_engines()
    fallback_name = get_settings().stt_fallback
    if fallback_name:
        FALLBACKS[fallback_name]()
