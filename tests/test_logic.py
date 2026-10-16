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


def test_rules_parse_hindi_and_urdu_commands():
    assert parse_with_rules("कृपया इस दस्तावेज का अंग्रेजी में अनुवाद कीजिये").target_language == "en"
    assert parse_with_rules("इसका हिंदी में अनुवाद करो").target_language == "hi"
    assert parse_with_rules("translate this into Bhojpuri").target_language == "bho"
    assert parse_with_rules("اس دستاویز کا اردو میں ترجمہ کریں").target_language == "ur"
    assert parse_with_rules("इसका भोजपुरी में अनुवाद करो").target_language == "bho"


def test_fidelity_accepts_numbers_in_other_digit_systems():
    src = "Interviews start on 15 July 2026; budget Rs 25,000."
    devanagari = "साक्षात्कार १५ जुलाई २०२६ को शुरू होंगे; बजट रु २५,००० है।"
    urdu = "انٹرویو ۱۵ جولائی ۲۰۲۶ کو شروع ہوں گے؛ بجٹ ۲۵٬۰۰۰ روپے۔"
    assert check_chunk(src, devanagari, same_language=False, min_chrf=0.8) == []
    assert check_chunk(src, urdu, same_language=False, min_chrf=0.8) == []
    assert any("numbers missing" in p for p in check_chunk(src, "साक्षात्कार जुलाई में", same_language=False, min_chrf=0.8))
