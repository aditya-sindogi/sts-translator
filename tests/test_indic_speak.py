from app.tts.indic_speak import IndicSpeakStreamer, split_clauses


def test_short_sentences_are_spoken_whole():
    assert split_clauses("I can't translate into Hindi yet.") == ["I can't translate into Hindi yet."]


def test_long_sentences_split_at_commas_without_tiny_fragments():
    urdu = "میں نے آپ کی دستاویز کا، جو چار صفحات پر مشتمل ہے، انگریزی سے اردو میں ترجمہ کر دیا ہے۔"
    parts = split_clauses(urdu)
    assert len(parts) == 2 and all(len(p) >= 25 for p in parts)
    assert " ".join(parts) == urdu
    english = "This agreement remains in force for two years, and either party may end it with thirty days' notice."
    assert split_clauses(english) == ["This agreement remains in force for two years,", "and either party may end it with thirty days' notice."]


def test_token_budget_scales_with_text_and_is_capped():
    short = IndicSpeakStreamer._token_budget("Stopped.")
    sentence = IndicSpeakStreamer._token_budget("x" * 80)
    assert 140 <= short < sentence <= 2520
    assert IndicSpeakStreamer._token_budget("x" * 5000) == 2520
