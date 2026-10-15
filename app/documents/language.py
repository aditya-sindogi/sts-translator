"""Written-language detection for extracted document text (lingua, Apache-2.0)."""

from collections import Counter
from functools import lru_cache

from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

# A bounded candidate set keeps memory and latency low. Extend when new languages ship.
_CANDIDATES = [
    Language.ENGLISH,
    Language.HINDI,
    Language.URDU,
    Language.BENGALI,
    Language.MARATHI,
    Language.PUNJABI,
    Language.ARABIC,
    Language.PERSIAN,
    Language.SPANISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.PORTUGUESE,
    Language.RUSSIAN,
    Language.CHINESE,
    Language.JAPANESE,
]


@lru_cache
def _detector():
    return LanguageDetectorBuilder.from_languages(*_CANDIDATES).build()


def detect_text_language(text: str) -> str | None:
    """ISO 639-1 code of the text's language, or None when undetermined."""
    sample = text[:3000]
    if len(sample.strip()) < 20:
        return None
    lang = _detector().detect_language_of(sample)
    return lang.iso_code_639_1.name.lower() if lang else None


def detect_document_language(page_texts: list[str]) -> tuple[str | None, dict[int, str]]:
    """Return (dominant language weighted by characters, per-page languages)."""
    per_page: dict[int, str] = {}
    weights: Counter[str] = Counter()
    for i, text in enumerate(page_texts, start=1):
        lang = detect_text_language(text)
        if lang:
            per_page[i] = lang
            weights[lang] += len(text)
    dominant = weights.most_common(1)[0][0] if weights else None
    return dominant, per_page


def language_name(code: str | None) -> str:
    if not code:
        return "an unknown language"
    try:
        return Language.from_iso_code_639_1(getattr(IsoCode639_1, code.upper())).name.title()
    except (AttributeError, ValueError):
        return code
