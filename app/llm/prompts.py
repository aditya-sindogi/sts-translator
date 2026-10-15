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
Rules:
- Preserve meaning, paragraph breaks and order.
- Keep numbers, dates, currencies, names and codes exactly as written.
- Do not summarize, explain, add or omit anything.
Output only the translation, with no preface."""

GIST_SYSTEM = """Summarize what the document is about in one or two plain sentences (at most {max_words} words), suitable to be spoken aloud.
Mention only topics that appear in the text. No lists, no markdown, no preface."""

CHUNK_GIST_SYSTEM = """In one short sentence, state the main topic of this excerpt. Use only information in the text. No preface."""
