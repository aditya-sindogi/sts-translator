import string

from app.session.assistant import _SENTENCE_END_RE, reply_language
from app.session.messages import MESSAGES, name_of, options, say
from app.translation.service import TranslationResult, TranslationService


def _fields(template: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def test_every_language_has_every_message_with_the_same_placeholders():
    english = MESSAGES["en"]
    for lang, table in MESSAGES.items():
        assert table.keys() == english.keys(), f"{lang} is missing or has extra messages"
        for key, template in table.items():
            assert _fields(template) == _fields(english[key]), f"{lang}.{key} placeholders differ"


def test_reply_in_spoken_language_only_with_messages_and_a_voice():
    tts = {"en": "kokoro:af_heart", "hi": "kokoro:hf_alpha"}
    assert reply_language("hi", tts) == "hi"
    assert reply_language("ur", tts) == "en"  # messages exist, but no Urdu voice configured
    assert reply_language("bho", {**tts, "bho": "x:y"}) == "en"  # voice, but no Bhojpuri messages
    assert reply_language("en", tts) == "en"


def test_language_names_and_lists_are_localized():
    assert name_of("en", "hi") == "अंग्रेज़ी"
    assert name_of("hi", "ur") == "ہندی"
    assert options(["en", "hi"], "hi") == "अंग्रेज़ी या हिंदी"
    assert options(["en"], "en") == "English"
    assert "अंग्रेज़ी" in say("ack", "hi", pages=4, target=name_of("en", "hi"), secs=10)


def test_summary_is_written_in_the_reply_language():
    svc = TranslationService.__new__(TranslationService)
    svc._settings = type("S", (), {"summary_max_words": 60})()
    r = TranslationResult("id", "doc", "f.pdf", "en", "en", 4, 1, "text", "", warnings=[])
    hindi = svc._compose_summary(r, "यह दस्तावेज़ रिमोट वर्क नीति के बारे में है।", "hi")
    assert hindi.startswith("मैंने आपके 4 पेज के दस्तावेज़ का अंग्रेज़ी से अंग्रेज़ी में अनुवाद कर दिया है।")
    assert hindi.endswith("पूरा अनुवाद आपकी स्क्रीन पर है।")
    assert svc._compose_summary(r, None, "en").startswith("I've translated your 4-page document from English to English.")


def test_sentences_split_on_devanagari_and_urdu_full_stops():
    assert len(_SENTENCE_END_RE.split("पहला वाक्य। दूसरा वाक्य।")) == 2
    assert len(_SENTENCE_END_RE.split("پہلا جملہ۔ دوسرا جملہ۔")) == 2
    assert len(_SENTENCE_END_RE.split("First. Second?")) == 2
