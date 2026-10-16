# Voice PDF Translator: how it works

You upload a PDF and say what you want, for example *"Translate this document into English."* You get back:
- **text:** the translated document plus a short summary of what was done
- **speech:** the summary, read aloud

The speech pipeline is built with **[Pipecat](https://github.com/pipecat-ai/pipecat)**. Gemma 4 does the language work, and the speech-to-text (STT) and text-to-speech (TTS) services can each be swapped through configuration.

---

## 1. The flow at a glance

```
           YOU                                          THE SERVER (one Pipecat pipeline per user)
 ┌────────────────────┐
 │ upload PDF ────────┼──► Document service: validate → extract text → detect language → split into chunks
 │                    │
 │ speak a command ───┼──► 1 Transport in   audio arrives from the browser (WebRTC)
 │                    │    2 VAD            "is the user speaking right now?"         (Silero)
 │                    │    3 Language ID    "which language is this?"                 (SpeechBrain)
 │                    │    4 STT            speech → text, engine chosen per language  (Parakeet, …)
 │                    │    5 Turn detection "has the user finished speaking?"         (Smart Turn)
 │                    │    6 Assistant      command → translate PDF → write summary    (Gemma 4)
 │                    │    7 TTS            summary text → speech                      (Kokoro, …)
 │ ◄── hear summary ──┼─── 8 Transport out  audio back to the browser
 │ ◄── see text ──────┼─── (step 6 also sends the translated text to the page)
 └────────────────────┘
```

### One request, step by step

Timings are measured on an M4 Pro Mac with the 2-page sample PDF.

| # | What happens | Done by | Time |
|---|---|---|---|
| 1 | You upload a PDF. It's checked: is it a real PDF, under 20 MB, not encrypted, not scanned? Its text is extracted, the language detected, and the text split into chunks of about 6,000 characters. | `app/documents/` | 0.02 s |
| 2 | You say *"Please translate this document into English."* | — | — |
| 3 | The VAD notices speech start and stop, so we know where the sentence begins and ends. | Pipecat + Silero | real time |
| 4 | Language ID listens to that sentence: **English, 99.6% sure**. | `app/lid/` | 0.015 s |
| 5 | STT sends the sentence to the engine configured for English (Parakeet): *"Please translate this document into English."* | `app/stt/` | 0.07 s |
| 6 | Turn detection decides you've finished talking, not just pausing. | Pipecat Smart Turn | ~0.5 s |
| 7 | The assistant works out what you want: **translate, target English**. Common phrasings are matched by simple rules; anything else goes to Gemma. | `app/commands/` | <0.01 s (rules), ~1 s (Gemma) |
| 8 | It speaks an acknowledgement right away: *"Translating your 2-page document into English…"* | Assistant → TTS | ~0.5 s to first audio |
| 9 | Gemma translates each chunk. Each chunk is checked: no numbers lost, length reasonable, and for English to English, close to the original wording. A chunk that fails is retried; if it fails again, the original text is kept and flagged. | `app/translation/` | ~5–15 s |
| 10 | The summary is built. The first sentence (*"I've translated your 2-page document from English to English."*) is generated from facts, not by the model. Gemma adds one or two sentences about what the document covers. | `app/translation/` | ~2 s |
| 11 | The translated text and summary go to the page, and the summary is spoken. | Assistant → page + TTS | — |

---

## 2. What Pipecat does

Pipecat is a framework for real-time voice pipelines. Its model is simple:

- **Frames** are small messages that flow through the pipeline: a 20 ms chunk of audio, "the user started speaking", a transcript, a line of text to speak, and so on.
- **Processors** receive frames, act on them and pass frames on. Each step in the diagram above is a processor.
- The **pipeline** is the list of processors, in order. The pipeline for this project is defined in [app/bot.py](app/bot.py) in about 10 lines.
- **Transports** connect the pipeline to the outside world. We use `SmallWebRTCTransport`: browser audio in and out, plus a data channel for text.
- **Services** are processors that wrap an AI model (STT, LLM, TTS). Pipecat ships many of them, and you can write your own.

**What Pipecat gives us for free:**

| Need | Pipecat piece |
|---|---|
| Browser audio in/out over WebRTC, plus a data channel for text | `SmallWebRTCTransport` |
| Detecting speech start and stop | `VADProcessor` + `SileroVADAnalyzer` |
| Deciding when the user has *finished* (not just paused) | `UserTurnProcessor` (Smart Turn model) |
| Stopping the bot's speech when the user talks over it | built-in interruption handling |
| Collecting each spoken sentence as one block of audio for STT | `SegmentedSTTService` (our STT is built on it) |
| Switching to a backup STT if the main one breaks | `ServiceSwitcher` with the failover strategy |
| Switching TTS voice per language | `ServiceSwitcher` with the manual strategy |
| Kokoro and Moonshine models | `KokoroTTSService`, `MoonshineSTTService` |
| Sending text events to the browser | the RTVI protocol (`RTVIServerMessageFrame`) |
| Timing metrics per step | built in (`enable_metrics=True`) |

**What we wrote on top:** language ID, STT routing, the document assistant, translation with checks, and the provider registries described in section 5.

---

## 3. How speech recognition (ASR/STT) works

```
audio ─► VAD marks one utterance ─► Language ID ─► choose engine for that language ─► transcript
                                       │                    │
                                       │                    └─ engine fails twice? ─► fallback STT (Moonshine, CPU)
                                       └─ confident it's a language with no engine? ─► "Please speak English"
```

1. **VAD (Silero)** detects where speech starts and stops. Pipecat buffers the audio in between as one segment.
2. **Language ID (SpeechBrain VoxLingua107)** classifies the segment's language (107 languages, about 15 ms on CPU). It runs *before* transcription, so we know which engine to use. It only rejects a sentence when it is confident (≥ 70%) that the language has no engine configured; short or uncertain sentences are let through.
3. **Routing** ([app/stt/routed.py](app/stt/routed.py)): `RoutedSTTService` looks up the engine for that language in `STS_STT` and sends it the audio (16 kHz, mono, 16-bit).
4. **The engine** returns text. A failure is retried once. A second failure makes Pipecat's `ServiceSwitcher` switch to the fallback service.
5. **The bot stops listening while it works.** From the moment it takes a command until it has finished speaking its answer, microphone audio is dropped, so background talk can't interrupt it. The page shows "busy, not listening". A short unclear fragment is the exception: the bot keeps listening for the rest of the command.
6. **Turn detection** groups the sentences into one turn. If you pause mid-command, the two halves are joined before the assistant sees them.

**Available engines:**

| Engine | Languages | Runs where | Notes |
|---|---|---|---|
| `parakeet` (NVIDIA Parakeet-TDT 0.6B v2) | English | Mac GPU (MLX) or NVIDIA GPU (NeMo) | Default. Most accurate on English commands; 0.07 s per sentence |
| `indic_zipformer` (team's k2-sherpa server) | 12 Indic languages + English | Remote GPU server (`STS_INDIC_ASR_URL`) | Hindi tested correct, ~1.7 s per sentence over the network. No Urdu |
| Moonshine (fallback) | English | CPU | Used automatically if the main STT fails |

---

## 4. How speech output (TTS) works

**The bot replies in the language you spoke.** Speak Hindi and its acknowledgement and spoken summary are in Hindi, even when the document is translated into English. This needs two things for that language:
- messages in [app/session/messages.py](app/session/messages.py) (English, Hindi and Urdu are there now)
- a voice in `STS_TTS`

If either is missing, it replies in English. Before speaking, the assistant switches to the voice configured for the reply language, and it sends one sentence at a time so audio starts sooner.

| Provider | Voices (examples) | Runs where | Notes |
|---|---|---|---|
| `kokoro` | `af_heart`, `am_michael` (English); `hf_alpha`, `hm_omega` (Hindi) | Local CPU | Default. Fast (~0.5 s to first audio). No Urdu |
| `indic_parler` (team's server) | `english_mary`, `hindi_rohit`, `urdu_divya`, … (23 languages) | Remote GPU (`STS_INDIC_TTS_URL`) | Natural voices, including Urdu, but slower than real time (~1.5×) |

---

## 5. Swapping STT and TTS

Each language maps to a provider in **one setting**, in `.env` or as an environment variable:

```bash
# Speech-to-text: spoken language -> engine
STS_STT='{"en": "parakeet"}'                                  # default
STS_STT='{"en": "parakeet", "hi": "indic_zipformer"}'         # also accept Hindi speech

# Text-to-speech: language of the text -> "provider:voice"
STS_TTS='{"en": "kokoro:af_heart"}'                           # default
STS_TTS='{"en": "indic_parler:english_mary", "hi": "kokoro:hf_alpha", "ur": "indic_parler:urdu_divya"}'
```

Restart the server and the new mapping is used. This exact swap was tested: Hindi speech went to Zipformer and English speech was spoken by Parler, with no code changes.

### Adding a new STT service (e.g. Wispr, Deepgram, Whisper)

Every STT service fits one small interface: *"here is one sentence of audio, give me the text"*.

```python
# app/stt/my_service.py
from app.stt.engines import STTEngine

class MySTTEngine(STTEngine):
    name = "my_service"
    languages = frozenset({"en", "hi"})          # languages it can transcribe

    async def transcribe(self, pcm16: bytes, language: str) -> str:
        # pcm16: one utterance, 16 kHz mono 16-bit. Call your model or API here.
        return text
```

```python
# app/stt/providers.py: register it
ENGINES = {
    ...,
    "my_service": lambda languages: MySTTEngine(),
}
```

```bash
STS_STT='{"en": "my_service"}'
```

[app/stt/indic_zipformer.py](app/stt/indic_zipformer.py) is a complete example of a remote WebSocket engine in about 80 lines.

### Adding a new TTS service (e.g. ElevenLabs, Cartesia, Piper)

Any Pipecat TTS service works as is. Register a factory that takes a voice and a language:

```python
# app/tts/providers.py
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

PROVIDERS = {
    ...,
    "elevenlabs": lambda voice, language: ElevenLabsTTSService(api_key=..., voice_id=voice),
}
```

```bash
STS_TTS='{"en": "elevenlabs:<voice-id>"}'
```

For a service Pipecat doesn't have, subclass `TTSService` and implement `run_tts(text)` to yield audio frames. [app/tts/indic_parler.py](app/tts/indic_parler.py) is a complete example in about 70 lines.

---

## 6. How a command becomes a translation

| Step | What happens | File |
|---|---|---|
| Understand the command | Simple rules catch *stop*, *repeat* and *"translate … into X"*. Anything else goes to Gemma, which must answer in a fixed JSON format: `{intent, target_language}`. If the target language is missing, the bot asks. | [app/commands/parser.py](app/commands/parser.py) |
| Check it can be done | Is a PDF uploaded and ready? Is the target language supported? Is another translation already running? | [app/session/assistant.py](app/session/assistant.py) |
| Translate | Chunks go to Gemma two at a time, with a prompt that says to keep numbers, names and structure. Progress is sent to the page. Saying "stop" cancels the job. | [app/translation/service.py](app/translation/service.py) |
| Check each chunk | No missing numbers, sensible length, and (same language) close to the source wording. On failure: retry at temperature 0, then keep the original and warn. | [app/translation/fidelity.py](app/translation/fidelity.py) |
| Summarise | The factual sentence comes from metadata. Gemma adds one or two sentences on the topic. | [app/translation/service.py](app/translation/service.py) |

Gemma is called through the standard OpenAI-compatible API ([app/llm/client.py](app/llm/client.py)), so any server that speaks it works: Ollama (current), vLLM, and others. Gemma 4's "thinking" mode is turned off, because it made answers about 10× slower with no benefit here.

---

## 7. Project layout

```
app/
  bot.py                 the Pipecat pipeline (the 8 steps above)
  server.py              web server: PDF upload, WebRTC connection, results
  config.py              every setting (env var = STS_ + name)
  documents/             PDF checks, text extraction, language detection, chunking
  lid/speech_lid.py      spoken language ID (pipeline step 3)
  stt/
    providers.py         ◄ STT registry: add/swap STT services here
    engines.py           the STT engine interface + Parakeet engine
    parakeet.py          Parakeet on Mac (MLX) or NVIDIA GPU (NeMo)
    indic_zipformer.py   team's Indic ASR server client
    routed.py            picks the engine per sentence (pipeline step 4)
  tts/
    providers.py         ◄ TTS registry: add/swap TTS services here
    indic_parler.py      team's Indic Parler-TTS client
  session/assistant.py   the "brain" (pipeline step 6)
  commands/parser.py     voice command → structured command
  translation/           chunked translation, fidelity checks, summary
  llm/                   Gemma client and prompts
web/index.html           test page: upload, microphone, results
scripts/                 tests without a browser, stack check
deploy/                  EC2 setup/start/stop (Ollama + app)
samples/                 test PDFs
tests/                   unit tests
```

---

## 8. Running it

**On a Mac:**
```bash
ollama pull gemma4:12b
uv sync
uv run python -m app.server          # then open http://127.0.0.1:7860
```

**On the EC2 GPU box** (Ollama runs Gemma on the GPU; nothing else is needed):
```bash
bash deploy/setup.sh                 # one time
bash deploy/start.sh                 # start Ollama + app   (bash deploy/stop.sh to stop)
# on your laptop:  ssh -N -L 7860:localhost:7860 l40-aws   then open http://localhost:7860
```
If the page connects but the bot never hears you, add a security-group rule: inbound UDP 1024–65535 from your IP.

**Checking it without a browser:**
```bash
uv run python -m scripts.check_stack                                     # PASS/FAIL per component
uv run python -m scripts.translate_cli samples/sample.pdf "translate this into English"
uv run python -m scripts.voice_e2e samples/sample.pdf "Please translate this document into English."
uv run pytest
```

---

## 9. Settings you're likely to change

| Setting | Default | Meaning |
|---|---|---|
| `STS_STT` | `{"en": "parakeet"}` | Spoken language → STT engine. These are also the languages accepted as commands |
| `STS_STT_FALLBACK` | `moonshine` | Backup STT if the main one fails (`""` = none) |
| `STS_TTS` | `{"en": "kokoro:af_heart"}` | Text language → `provider:voice` |
| `STS_SUPPORTED_LANGUAGES` | `["en", "hi", "ur", "bho"]` | Languages a document can be in, and be translated into |
| `STS_LLM_BASE_URL` / `STS_LLM_MODEL` | Ollama, `gemma4:12b` | Which LLM server and model |
| `STS_PARAKEET_BACKEND` | `auto` | `mlx` (Mac), `nemo` (NVIDIA), or `auto` |
| `STS_INDIC_ASR_URL` / `STS_INDIC_TTS_URL` | team servers | Addresses of the team's Indic ASR / TTS |

All settings are in [app/config.py](app/config.py).

## 10. Current limits

- **Translation targets:** English, Hindi, Urdu and Bhojpuri (`STS_SUPPORTED_LANGUAGES`). All three were tested on the sample policy PDF. Hindi and Bhojpuri read naturally. Urdu sometimes keeps country names in Latin letters. Gemma has no official Bhojpuri support, so have a native speaker review Bhojpuri output.
- **No OCR yet:** scanned PDFs (no text layer) are rejected.
- **Kokoro licensing:** Kokoro pulls in GPL components (espeak-ng). Review this before any commercial use.
