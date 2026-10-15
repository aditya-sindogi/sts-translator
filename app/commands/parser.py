"""Turn a spoken transcript into a structured command.

A fast rule layer handles stop/repeat and obvious "translate into X" phrasing so those
work even when the LLM is slow or down. Everything else goes to Gemma with a JSON schema.
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from loguru import logger

from app.llm.client import LLMUnavailable, get_llm
from app.llm.prompts import COMMAND_SCHEMA, COMMAND_SYSTEM


class Intent(StrEnum):
    TRANSLATE = "translate"
    REPEAT_SUMMARY = "repeat_summary"
    STOP = "stop"
    HELP = "help"
    UNCLEAR = "unclear"


@dataclass
class Command:
    intent: Intent
    target_language: str | None = None
    source: str = "rules"  # "rules" | "llm"


# Spoken language names -> ISO 639-1. Extend as languages are added.
LANGUAGE_NAMES = {
    "english": "en",
    "hindi": "hi",
    "urdu": "ur",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "arabic": "ar",
    "bengali": "bn",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
    "chinese": "zh",
    "japanese": "ja",
}

_STOP_RE = re.compile(r"^\W*(stop|cancel|abort|never ?mind|quit)\b", re.IGNORECASE)
_REPEAT_RE = re.compile(r"\b(repeat|say (that|it) again|again please|what did you say)\b", re.IGNORECASE)
_TRANSLATE_RE = re.compile(r"\btranslat\w*\b", re.IGNORECASE)
_TARGET_RE = re.compile(r"\b(?:in|into|to)\s+([a-z]+)\b", re.IGNORECASE)


def normalize_language(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    if v in LANGUAGE_NAMES.values():
        return v
    return LANGUAGE_NAMES.get(v)


def parse_with_rules(text: str) -> Command | None:
    if _STOP_RE.search(text):
        return Command(Intent.STOP)
    if _REPEAT_RE.search(text):
        return Command(Intent.REPEAT_SUMMARY)
    if _TRANSLATE_RE.search(text):
        for match in _TARGET_RE.finditer(text):
            lang = normalize_language(match.group(1))
            if lang:
                return Command(Intent.TRANSLATE, lang)
    return None


async def parse_command(text: str) -> Command:
    text = text.strip()
    if not text:
        return Command(Intent.UNCLEAR)

    ruled = parse_with_rules(text)
    if ruled:
        return ruled

    try:
        raw = await get_llm().complete(
            [{"role": "system", "content": COMMAND_SYSTEM}, {"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=100,
            json_schema=COMMAND_SCHEMA,
        )
        data = json.loads(raw)
        intent = Intent(data.get("intent", "unclear"))
        target = normalize_language(data.get("target_language"))
        if data.get("target_language") and not target:
            logger.warning(f"unrecognized target language from LLM: {data.get('target_language')!r}")
        return Command(intent, target, source="llm")
    except LLMUnavailable:
        logger.warning("LLM unavailable for command parsing; falling back to rules")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"could not parse LLM command output: {e}")

    # Degraded mode: a bare "translate" with no language still counts.
    if _TRANSLATE_RE.search(text):
        return Command(Intent.TRANSLATE, None)
    return Command(Intent.UNCLEAR)
