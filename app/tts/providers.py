"""Text-to-speech providers: the one place to add or swap a TTS service.

``STS_TTS`` maps the language of the text being spoken to ``"provider:voice"``:

    STS_TTS='{"en": "kokoro:af_heart"}'                                       # default
    STS_TTS='{"en": "kokoro:af_heart", "hi": "kokoro:hf_alpha", "ur": "indic_parler:urdu_divya"}'

Before speaking, the assistant switches to the service for that text's language.

To add a provider (e.g. ElevenLabs, Cartesia, Piper, any Pipecat TTS service):
  1. Register a factory in ``PROVIDERS``: ``(voice, language) -> Pipecat TTSService``.
     Use a built-in Pipecat service, or write your own ``TTSService`` (see indic_parler.py).
  2. Point a language at ``"<provider>:<voice>"`` in ``STS_TTS``.
"""

from collections.abc import Callable

from loguru import logger

from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyManual
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, KokoroTTSService, _ensure_model_files
from pipecat.transcriptions.language import Language

from app.config import get_settings
from app.tts.indic_parler import IndicParlerTTSService


def _kokoro(voice: str, language: str) -> FrameProcessor:
    return KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=voice, language=Language(language)),
        stop_frame_timeout_s=8.0,  # CPU synthesis of a long sentence can exceed the 3 s default
    )


def _indic_parler(voice: str, language: str) -> FrameProcessor:
    return IndicParlerTTSService(
        base_url=get_settings().indic_tts_url,
        voice=voice,
        stop_frame_timeout_s=15.0,  # remote GPU synthesis runs ~1.5x real time
    )


# Provider name -> factory(voice, language) -> Pipecat TTS service
PROVIDERS: dict[str, Callable[[str, str], FrameProcessor]] = {
    "kokoro": _kokoro,
    "indic_parler": _indic_parler,
}

TTSSelector = Callable[[str], FrameProcessor | None]


def _parse(spec: str) -> tuple[str, str]:
    provider, _, voice = spec.partition(":")
    if provider not in PROVIDERS or not voice:
        raise ValueError(f"STS_TTS: expected 'provider:voice' with provider in {sorted(PROVIDERS)}, got {spec!r}")
    return provider, voice


def build_tts_service() -> tuple[FrameProcessor, TTSSelector]:
    """The TTS stage of the pipeline, plus a selector: text language -> service to switch to."""
    by_language: dict[str, FrameProcessor] = {}
    by_spec: dict[str, FrameProcessor] = {}  # languages sharing a provider:voice share a service
    for lang, spec in get_settings().tts.items():
        if spec not in by_spec:
            provider, voice = _parse(spec)
            by_spec[spec] = PROVIDERS[provider](voice, lang)
        by_language[lang] = by_spec[spec]

    default = by_language.get("en") or next(iter(by_language.values()))
    services = list(by_spec.values())
    if len(services) == 1:
        return services[0], lambda lang: None

    logger.info(f"TTS: {get_settings().tts}")
    switcher = ServiceSwitcher(services=services, strategy_type=ServiceSwitcherStrategyManual)
    return switcher, lambda lang: by_language.get(lang, default)


def warm_up() -> None:
    """Download model files for local providers before the first caller arrives."""
    if any(spec.startswith("kokoro:") for spec in get_settings().tts.values()):
        _ensure_model_files(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", KOKORO_CACHE_DIR / "voices-v1.0.bin")
