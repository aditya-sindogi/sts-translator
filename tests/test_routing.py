import pytest

from app.lid.speech_lid import LanguageDetectedFrame
from app.stt.engines import STTEngine
from app.stt.indic_zipformer import IndicZipformerEngine
from app.stt.routed import RoutedSTTService


class FakeEngine(STTEngine):
    def __init__(self, name, languages):
        self.name = name
        self.languages = frozenset(languages)

    async def transcribe(self, pcm16, language):
        return f"{self.name}:{language}"


def _lid(lang, conf):
    return LanguageDetectedFrame(language=lang, confidence=conf, duration_secs=2.0, top=[])


def test_routes_by_detected_language_with_confidence_gate():
    stt = RoutedSTTService(engines=[FakeEngine("parakeet", ["en"]), FakeEngine("zipformer", ["hi", "en"])])
    assert stt._routes["en"].name == "parakeet"  # first engine listed wins
    assert stt._routes["hi"].name == "zipformer"
    assert stt.language_for(_lid("hi", 0.9)) == "hi"
    assert stt.language_for(_lid("hi", 0.3)) == "en"  # not confident -> default
    assert stt.language_for(_lid("ur", 0.99)) == "en"  # no engine for Urdu -> default
    assert stt.language_for(_lid(None, 0.0)) == "en"  # segment too short for LID


def test_default_language_requires_an_engine():
    with pytest.raises(ValueError):
        RoutedSTTService(engines=[FakeEngine("zipformer", ["hi"])], default_language="en")


def test_zipformer_rejects_languages_without_models():
    with pytest.raises(ValueError, match="ur"):
        IndicZipformerEngine("ws://unused", ["hi", "ur"])
