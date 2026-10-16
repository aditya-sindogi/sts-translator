import pytest

from app.config import get_settings
from app.lid.speech_lid import LanguageDetectedFrame
from app.stt import providers as stt_providers
from app.stt.engines import STTEngine
from app.stt.indic_zipformer import IndicZipformerEngine
from app.stt.routed import RoutedSTTService
from app.tts import providers as tts_providers


class FakeEngine(STTEngine):
    def __init__(self, name, languages):
        self.name = name
        self.languages = frozenset(languages)

    async def transcribe(self, pcm16, language):
        return f"{self.name}:{language}"


def _lid(lang, conf):
    return LanguageDetectedFrame(language=lang, confidence=conf, duration_secs=2.0, top=[])


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "stt", dict(s.stt))
    monkeypatch.setattr(s, "tts", dict(s.tts))
    return s


def test_routes_by_detected_language_with_confidence_gate():
    en, hi = FakeEngine("parakeet", ["en"]), FakeEngine("zipformer", ["hi"])
    stt = RoutedSTTService(routes={"en": en, "hi": hi})
    assert stt.language_for(_lid("hi", 0.9)) == "hi"
    assert stt.language_for(_lid("hi", 0.3)) == "en"  # not confident -> default
    assert stt.language_for(_lid("ur", 0.99)) == "en"  # no engine for Urdu -> default
    assert stt.language_for(_lid(None, 0.0)) == "en"  # segment too short for LID


def test_stt_config_maps_languages_to_registered_engines(settings, monkeypatch):
    monkeypatch.setitem(stt_providers.ENGINES, "fake_en", lambda langs: FakeEngine("fake_en", ["en"]))
    monkeypatch.setitem(stt_providers.ENGINES, "fake_multi", lambda langs: FakeEngine("fake_multi", langs))
    settings.stt = {"en": "fake_en", "hi": "fake_multi", "bn": "fake_multi"}
    routes = stt_providers.build_engines()
    assert routes["en"].name == "fake_en"
    assert routes["hi"] is routes["bn"]  # one engine instance serves both languages


def test_stt_config_rejects_unknown_engine_and_unsupported_language(settings, monkeypatch):
    settings.stt = {"en": "nope"}
    with pytest.raises(ValueError, match="unknown STT engine"):
        stt_providers.build_engines()
    monkeypatch.setitem(stt_providers.ENGINES, "fake_en", lambda langs: FakeEngine("fake_en", ["en"]))
    settings.stt = {"hi": "fake_en"}
    with pytest.raises(ValueError, match="can't transcribe"):
        stt_providers.build_engines()


def test_stt_engine_that_fails_to_load_is_skipped(settings, monkeypatch):
    def broken(langs):
        raise RuntimeError("no GPU")

    monkeypatch.setitem(stt_providers.ENGINES, "broken", broken)
    settings.stt = {"en": "broken"}
    assert stt_providers.build_engines() == {}


def test_tts_config_picks_provider_and_voice_per_language(settings, monkeypatch):
    made = []
    monkeypatch.setitem(
        tts_providers.PROVIDERS, "fake", lambda voice, lang: made.append((voice, lang)) or f"svc:{voice}"
    )
    monkeypatch.setattr(tts_providers, "ServiceSwitcher", lambda services, strategy_type: ("switcher", services))
    settings.tts = {"en": "fake:mary", "hi": "fake:rohit", "mr": "fake:rohit"}
    switcher, select = tts_providers.build_tts_service()
    assert made == [("mary", "en"), ("rohit", "hi")]  # same provider:voice is built once
    assert select("hi") == select("mr") == "svc:rohit"
    assert select("ta") == "svc:mary"  # no voice for the language -> English voice


def test_tts_config_rejects_bad_spec(settings):
    settings.tts = {"en": "kokoro"}
    with pytest.raises(ValueError, match="provider:voice"):
        tts_providers.build_tts_service()


def test_zipformer_rejects_languages_without_models():
    with pytest.raises(ValueError, match="ur"):
        IndicZipformerEngine("ws://unused", ["hi", "ur"])
