from app.commands.parser import Intent, normalize_language, parse_with_rules
from app.translation.fidelity import check_chunk, chrf

SRC = "The Q3 budget is $4.2M, up 12% from Q2. Hiring starts on 15 March 2026."


def test_chrf_identity_and_divergence():
    assert chrf(SRC, SRC) == 1.0
    assert chrf("Completely unrelated sentence about cats.", SRC) < 0.4


def test_fidelity_passes_faithful_copy():
    assert check_chunk(SRC, SRC, same_language=True, min_chrf=0.8) == []


def test_fidelity_flags_dropped_numbers_and_paraphrase():
    para = "The third-quarter budget is 4.2 million dollars, a rise from the second quarter."
    problems = check_chunk(SRC, para, same_language=True, min_chrf=0.8)
    assert any("numbers missing" in p for p in problems)
    assert any("chrF" in p for p in problems)


def test_fidelity_flags_truncation():
    problems = check_chunk(SRC * 5, SRC, same_language=True, min_chrf=0.8)
    assert any("length ratio" in p for p in problems)


def test_fidelity_cross_language_skips_chrf():
    out = "तीसरी तिमाही का बजट $4.2M है, Q2 से 12% अधिक। भर्ती 15 मार्च 2026 को शुरू होगी। Q3"
    assert check_chunk(SRC, out, same_language=False, min_chrf=0.8) == []


def test_rules_parse_common_commands():
    cmd = parse_with_rules("Please translate this document into English")
    assert cmd.intent == Intent.TRANSLATE and cmd.target_language == "en"
    assert parse_with_rules("translate it to hindi").target_language == "hi"
    assert parse_with_rules("stop").intent == Intent.STOP
    assert parse_with_rules("can you repeat that").intent == Intent.REPEAT_SUMMARY
    assert parse_with_rules("translate this") is None  # no language: defer to the LLM / clarify
    assert parse_with_rules("what's the weather") is None


def test_normalize_language():
    assert normalize_language("English") == "en"
    assert normalize_language("ur") == "ur"
    assert normalize_language("Klingon") is None
    assert normalize_language(None) is None
