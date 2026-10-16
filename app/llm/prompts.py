"""Prompt templates. Kept in one place so they can be versioned and evaluated."""

COMMAND_SYSTEM = """You parse spoken voice commands for a document translation assistant.
The user has uploaded (or will upload) a PDF. Classify the command:
- "translate": the user wants the document translated. Extract the target language as an ISO 639-1 code (e.g. "en", "hi", "ur"). If the user does not name a target language, set target_language to null.
- "repeat_summary": the user wants to hear the last result again.
- "stop": the user wants to cancel or stop the current work.
- "help": the user asks what you can do.
- "unclear": anything else, or speech that is not a command.
Transcripts come from speech recognition and may contain small errors; infer the most likely meaning but never guess a language that was not mentioned."""

COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["translate", "repeat_summary", "stop", "help", "unclear"]},
        "target_language": {"type": ["string", "null"]},
    },
    "required": ["intent", "target_language"],
}

TRANSLATE_SAME_LANG_SYSTEM = """You are a precise document translator. The source text is already in {target_name}.
Reproduce it faithfully in {target_name}. Rules:
- Keep the original wording, sentence order and paragraph breaks.
- Keep every number, date, currency, abbreviation, name and code exactly as written.
- Only fix obvious text-extraction artifacts (broken words, stray symbols).
- Do not summarize, explain, add or omit anything.
Output only the text, with no preface."""

TRANSLATE_SYSTEM = """You are a precise document translator. Translate the text from {source_name} into {target_name}.
{target_note}
Rules:
- Translate every line, including titles, headings and section labels.
- Preserve meaning, paragraph breaks and order.
- Write place and country names with their standard name in the target language; never substitute a different place.
- Keep numbers, dates, currencies, person and company names, and codes exactly as written.
- Do not summarize, explain, add or omit anything.
Output only the translation, with no preface."""

GIST_SYSTEM = """Summarize what the document is about in one or two plain sentences (at most {max_words} words), written in {language}, suitable to be spoken aloud.
Mention only topics that appear in the text. Write numbers as words. No lists, no markdown, no preface."""

CHUNK_GIST_SYSTEM = """In one short sentence, state the main topic of this excerpt. Use only information in the text. No preface."""

# How to write each target language (script and variety). Empty for languages that need no note.
TARGET_NOTES = {
    "hi": "Write standard Hindi in Devanagari script (India → भारत, United States → संयुक्त राज्य अमेरिका). Keep common English technical terms (VPN, IT, PDF) as they are.",
    "ur": "Write Urdu in the Perso-Arabic (Nastaliq) script, not Devanagari (India → بھارت, United States → امریکہ, Singapore → سنگاپور). Keep common English technical terms (VPN, IT, PDF) in Latin letters.",
    "bho": "Write Bhojpuri, not Hindi: use Bhojpuri grammar and vocabulary (for example बा/बाड़े/रहल, हमनी, ओकरा) in Devanagari script (India → भारत, United States → अमेरिका).",
}

# Re-translates lines a chunk translation left in the source language (often short headings).
REPAIR_LINES_SYSTEM = """Translate each line in the JSON list from {source_name} into {target_name}.
{target_note}
These lines were left untranslated, usually titles or headings: translate their words.
Only a person's or company's own name stays as written (e.g. "Northwind Analytics"); everything else is translated.
Keep list numbering such as "1." and numbers and dates as written.
Return JSON {{"lines": [...]}} with exactly as many lines as given, in the same order."""

REPAIR_LINES_SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
    "required": ["lines"],
}
