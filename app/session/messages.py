"""Everything the bot says, per reply language.

The bot replies in the language the user spoke, when that language has messages here and
a TTS voice in STS_TTS; otherwise it replies in English. To add a reply language, add a
block with the same keys (a test checks every block is complete).
"""

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "ack": "Translating your {pages}-page document into {target}. This may take about {secs} seconds.",
        "ask_target": "Which language should I translate the document into? You can choose {options}.",
        "busy": "I'm already translating. Say stop to cancel.",
        "doc_processing": "Your PDF is still being processed. Please ask again in a moment.",
        "doc_rejected": "I couldn't use that PDF. {reason}",
        "error": "Sorry, the translation failed. Please try again.",
        "help": "Upload a PDF, then say something like: translate this document into English.",
        "no_document": "Please upload a PDF first, then ask me again.",
        "nothing_to_repeat": "There is nothing to repeat yet.",
        "nothing_to_stop": "There's nothing to stop.",
        "stopped": "Stopped.",
        "target_unsupported": "I can't translate into {target} yet. You can choose {options}.",
        "unclear": "Sorry, I didn't catch a command. You can say: translate this document into English.",
        "summary_done": "I've translated your {pages}-page document from {source} to {target}.",
        "summary_kept": "{count} section(s) could not be verified and were left as written.",
        "summary_screen": "The full text is on your screen.",
    },
    "hi": {
        "ack": "आपके {pages} पेज के दस्तावेज़ का {target} में अनुवाद हो रहा है। इसमें लगभग {secs} सेकंड लग सकते हैं।",
        "ask_target": "दस्तावेज़ का अनुवाद किस भाषा में करूँ? आप {options} चुन सकते हैं।",
        "busy": "मैं पहले से अनुवाद कर रहा हूँ। रोकने के लिए स्टॉप कहें।",
        "doc_processing": "आपकी PDF अभी तैयार हो रही है। कृपया थोड़ी देर बाद फिर से कहें।",
        "doc_rejected": "मैं इस PDF का इस्तेमाल नहीं कर सका। {reason}",
        "error": "माफ़ कीजिए, अनुवाद नहीं हो सका। कृपया फिर से कोशिश करें।",
        "help": "एक PDF अपलोड करें, फिर कहें: इस दस्तावेज़ का अंग्रेज़ी में अनुवाद करो।",
        "no_document": "कृपया पहले एक PDF अपलोड करें, फिर दोबारा कहें।",
        "nothing_to_repeat": "अभी दोहराने के लिए कुछ नहीं है।",
        "nothing_to_stop": "रोकने के लिए कुछ नहीं चल रहा है।",
        "stopped": "रोक दिया।",
        "target_unsupported": "मैं अभी {target} में अनुवाद नहीं कर सकता। आप {options} चुन सकते हैं।",
        "unclear": "माफ़ कीजिए, मैं समझ नहीं पाया। आप कह सकते हैं: इस दस्तावेज़ का अंग्रेज़ी में अनुवाद करो।",
        "summary_done": "मैंने आपके {pages} पेज के दस्तावेज़ का {source} से {target} में अनुवाद कर दिया है।",
        "summary_kept": "{count} हिस्सों की जाँच नहीं हो सकी, इसलिए उन्हें मूल रूप में छोड़ा गया है।",
        "summary_screen": "पूरा अनुवाद आपकी स्क्रीन पर है।",
    },
    "ur": {
        "ack": "آپ کی {pages} صفحات کی دستاویز کا {target} میں ترجمہ ہو رہا ہے۔ اس میں تقریباً {secs} سیکنڈ لگ سکتے ہیں۔",
        "ask_target": "دستاویز کا ترجمہ کس زبان میں کروں؟ آپ {options} منتخب کر سکتے ہیں۔",
        "busy": "میں پہلے ہی ترجمہ کر رہا ہوں۔ روکنے کے لیے اسٹاپ کہیں۔",
        "doc_processing": "آپ کی PDF ابھی تیار ہو رہی ہے۔ براہ کرم تھوڑی دیر بعد دوبارہ کہیں۔",
        "doc_rejected": "میں اس PDF کو استعمال نہیں کر سکا۔ {reason}",
        "error": "معاف کیجیے، ترجمہ نہیں ہو سکا۔ براہ کرم دوبارہ کوشش کریں۔",
        "help": "ایک PDF اپ لوڈ کریں، پھر کہیں: اس دستاویز کا انگریزی میں ترجمہ کرو۔",
        "no_document": "براہ کرم پہلے ایک PDF اپ لوڈ کریں، پھر دوبارہ کہیں۔",
        "nothing_to_repeat": "ابھی دہرانے کے لیے کچھ نہیں ہے۔",
        "nothing_to_stop": "روکنے کے لیے کچھ نہیں چل رہا۔",
        "stopped": "روک دیا۔",
        "target_unsupported": "میں ابھی {target} میں ترجمہ نہیں کر سکتا۔ آپ {options} منتخب کر سکتے ہیں۔",
        "unclear": "معاف کیجیے، میں سمجھ نہیں سکا۔ آپ کہہ سکتے ہیں: اس دستاویز کا انگریزی میں ترجمہ کرو۔",
        "summary_done": "میں نے آپ کی {pages} صفحات کی دستاویز کا {source} سے {target} میں ترجمہ کر دیا ہے۔",
        "summary_kept": "{count} حصوں کی جانچ نہیں ہو سکی، اس لیے انہیں اصل حالت میں چھوڑ دیا گیا ہے۔",
        "summary_screen": "پورا ترجمہ آپ کی اسکرین پر ہے۔",
    },
}

# Language names as spoken in each reply language.
LANGUAGE_NAMES: dict[str, dict[str, str]] = {
    "en": {"en": "English", "hi": "Hindi", "ur": "Urdu", "bho": "Bhojpuri", "awa": "Awadhi"},
    "hi": {"en": "अंग्रेज़ी", "hi": "हिंदी", "ur": "उर्दू", "bho": "भोजपुरी", "awa": "अवधी"},
    "ur": {"en": "انگریزی", "hi": "ہندی", "ur": "اردو", "bho": "بھوجپوری", "awa": "اودھی"},
}

_OR = {"en": " or ", "hi": " या ", "ur": " یا "}


def say(key: str, language: str, **values) -> str:
    """The message ``key`` in ``language`` (English if that language has no messages)."""
    table = MESSAGES.get(language, MESSAGES["en"])
    return table[key].format(**values)


def name_of(code: str | None, language: str) -> str:
    """How a language is named when speaking ``language`` (falls back to English names)."""
    from app.documents.language import language_name

    names = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES["en"])
    return names.get(code or "", None) or LANGUAGE_NAMES["en"].get(code or "", None) or language_name(code)


def options(codes: list[str], language: str) -> str:
    """'English or Hindi' style list of languages, in ``language``."""
    names = [name_of(c, language) for c in codes]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + _OR.get(language, " or ") + names[-1]
