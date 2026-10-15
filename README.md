# Voice PDF Translator (Phase 1, English)

You speak a command and upload a PDF. You get back the translated text plus a short summary, and the summary is also read aloud.

```
mic ─► Silero VAD ─► Spoken language ID ─► STT: Parakeet (⇄ Moonshine failover) ─► Smart-turn
    ─► DocumentAssistant: command → chunked Gemma 4 translation → grounded summary
    ─► Kokoro TTS ─► speaker
       └─► text result (summary, translated text, warnings) over the WebRTC data channel
```

Everything runs locally, and no audio or document leaves the machine. Wispr is planned as an optional cloud STT provider once API access is granted.

## Quick start (macOS, Apple Silicon)

```bash
ollama pull gemma4:12b                 # LLM, served by Ollama's OpenAI-compatible API
uv sync                                # Python 3.12 environment
uv run python -m app.server            # first run downloads Parakeet, LID, Moonshine and Kokoro models
open http://127.0.0.1:7860
```

In the page:
1. Upload a PDF.
2. Click **Connect microphone**.
3. Say *"Translate this document into English."*

Sample PDFs are in `samples/`: `mock_remote_work_policy.pdf` (4 pages) and `sample.pdf` (2 pages).

## Deploy on EC2 (GPU)

On EC2, Gemma 4 12B runs in **vLLM on the instance's GPU**, not in Ollama, and Parakeet runs on the GPU through NeMo. No sudo or Docker is needed.

**Target instance:** L40S (46 GB), Ubuntu 22.04, with `uv` and Python 3.12 installed. Setup needs about 45 GB of free disk.

```bash
# on the EC2
git clone <your-repo-url> sts-up && cd sts-up
bash deploy/setup.sh                       # one time: envs, vLLM nightly, model downloads (~20-40 min)
bash deploy/start.sh                       # starts vLLM (Gemma) and then the app; logs in logs/
uv run python -m scripts.check_stack       # PASS/FAIL for LLM, thinking off, Parakeet, LID, TTS
```

```bash
# on your laptop
ssh -N -L 7860:localhost:7860 l40-aws      # keep this running
open http://localhost:7860                 # mic works because the page is on localhost
```

What `setup.sh` does:
- Writes `.env` from [deploy/ec2.env.example](deploy/ec2.env.example) and generates a vLLM API key.
- Runs `uv sync --extra nemo`.
- Installs a separate vLLM nightly environment in `~/.venvs/vllm`. Gemma 4 12B isn't in a stable vLLM release yet.
- Pre-downloads all the models.

How it runs:
- vLLM listens on `127.0.0.1:8000` and uses 70% of GPU memory, leaving room for Parakeet.
- The app listens on `127.0.0.1:7860`, so it's only reachable through the SSH tunnel.
- `STS_LLM_REQUIRED=true` makes the app refuse to start if vLLM isn't up. It never silently falls back.
- Stop everything with `bash deploy/stop.sh` (or `stop.sh app` / `stop.sh vllm`).

**WebRTC audio:** the page loads through the tunnel, but the audio itself goes over UDP straight to the instance's public IP, which STUN lets it advertise. This usually works with no AWS changes. If the page connects but audio never arrives, add a security-group inbound rule: **Custom UDP, ports 1024–65535, source = your IP/32**.

## Testing without a browser

```bash
uv run pytest                                                       # unit tests
uv run python -m scripts.check_stack                                # PASS/FAIL per component
uv run python -m scripts.translate_cli samples/sample.pdf "translate this into English"   # no voice
uv run python -m scripts.voice_e2e samples/sample.pdf "Please translate this document into English."
uv run python -m scripts.voice_e2e samples/sample.pdf "कृपया इस दस्तावेज़ का अंग्रेज़ी में अनुवाद कीजिए" hf_alpha hi
```

`voice_e2e` is a headless WebRTC client. It synthesizes the spoken command with Kokoro and connects to the server the same way the browser does. It then prints every event the bot sends back. The Hindi example checks that the language-ID step rejects non-English speech.

## How Pipecat is used

Pipecat moves audio and events between the stages as *frames*, and each stage is a *processor* in one pipeline per session. The stages are VAD, language ID, STT, turn detection, the assistant and TTS. Pipecat provides the realtime plumbing, so this code only adds the parts specific to this product.

| Concern | Provided by Pipecat | Written here |
|---|---|---|
| WebRTC audio in/out, signalling, data channel | `SmallWebRTCTransport`, `SmallWebRTCRequestHandler` | `/api/offer` endpoint |
| Voice activity detection | `VADProcessor` + `SileroVADAnalyzer` | — |
| End-of-turn detection, barge-in | `UserTurnProcessor` (Smart Turn v3), interruptions | — |
| Segmenting speech for STT | `SegmentedSTTService` (buffers each utterance, queues it, runs off the audio path) | `RoutedSTTService`: picks an engine per utterance from the detected language (Parakeet on MLX/NeMo; optional Indic Zipformer) |
| STT failover | `ServiceSwitcher` + `ServiceSwitcherStrategyFailover`, built-in `MoonshineSTTService` | Parakeet reports a permanent error after a retry, which triggers the switch |
| Spoken language ID | `FrameProcessor` base class, custom frames | `LanguageIDProcessor` and `LanguageDetectedFrame` (SpeechBrain ECAPA) |
| TTS | `KokoroTTSService`, `TTSService` base, manual `ServiceSwitcher` | `IndicParlerTTSService`; switches voice by text language; one request per sentence |
| Text to the client | RTVI protocol (`RTVIServerMessageFrame`, automatic `user-transcription` events) | Message types: `language`, `command`, `assistant`, `progress`, `result` |
| Metrics / tracing | TTFB and processing metrics per service, OpenTelemetry hooks (`traced_stt`) | — |

`DocumentAssistant` ([app/session/assistant.py](app/session/assistant.py)) works as follows:
- It collects each turn's transcript and checks the detected language.
- It parses the command, first with fast rules and then with Gemma using a JSON schema.
- It runs the translation as a background task, so the voice loop stays responsive and "stop" works mid-job.
- It sends two outputs: an RTVI text message, and `TTSSpeakFrame`s for Kokoro.

Gemma is called directly through the OpenAI-compatible API ([app/llm/client.py](app/llm/client.py)) instead of through Pipecat's conversational LLM service. Translation is a batch job made of chunked calls with checks on each chunk, not a streaming chat. Swapping Ollama for vLLM only changes `STS_LLM_BASE_URL`.

## Safeguards built in

- **Spoken language:** a turn is rejected only when the language-ID model is at least 70% confident it is not English. Short or uncertain segments are let through, because falsely rejecting accented English is worse.
- **PDF handling:**
  - Uploads are checked for size, PDF magic bytes, encryption, page limit, a missing text layer (scanned PDFs are rejected), and document language.
  - Paragraphs are rebuilt from line geometry.
  - Repeated headers, footers and page numbers are removed.
- **Translation fidelity:** every chunk is checked for numbers that went missing, its length ratio, and its chrF score against the source (for same-language output). A chunk that fails is retried once at temperature 0. If it fails again, the original text is kept and a warning is attached.
- **Summary:** the sentence saying what was done ("I've translated your N-page document from X to Y") is built from metadata. Only the description of the content comes from Gemma.
- **Failure handling:**
  - LLM: retries, timeouts and a circuit breaker.
  - STT: failover from Parakeet to Moonshine.
  - Language ID: an error never breaks the voice loop.
  - Text output is delivered even if TTS fails.

## Measured on an M4 Pro (24 GB), with models warm

| Step | Time |
|---|---|
| Parakeet (MLX), 3 s utterance | ~0.07 s |
| Language ID (CPU) | ~0.015 s |
| Transcript → spoken acknowledgement starts | ~0.5 s |
| Gemma 4 12B (Ollama, thinking off), 2-page translation + summary | ~5 s |
| End of speech → acknowledgement audio | ~2.5 s (mostly turn detection; tuning not done yet) |

Gemma 4 "thinks" by default, which made answers about 10× slower in testing. Ollama switches this off with `reasoning_effort: "none"` (`STS_LLM_EXTRA_BODY`). Check the equivalent flag when moving to vLLM.

## Optional: the team's Indic ASR and TTS servers

Two providers from `voice-models/` plug in behind the same interfaces. Both are switched off by default.

| Provider | What | How it's wired |
|---|---|---|
| Indic Zipformer ASR (`ws://…:6000`) | k2-sherpa streaming Zipformer on GPU, 12 Indic languages + English | `IndicZipformerEngine` ([app/stt/indic_zipformer.py](app/stt/indic_zipformer.py)). `RoutedSTTService` sends each utterance to it when language ID detects a language in `STS_INDIC_ASR_LANGUAGES`. English stays on Parakeet. |
| Indic Parler-TTS (`http://…:6001`) | ai4bharat/indic-parler-tts on GPU, 23 languages incl. **Hindi and Urdu** | `IndicParlerTTSService` ([app/tts/indic_parler.py](app/tts/indic_parler.py)), one service per language in a manual `ServiceSwitcher`. The assistant switches to the right voice before speaking text in that language. |

```bash
STS_INDIC_ASR_URL=ws://52.6.34.53:6000 \
STS_INDIC_TTS_URL=http://52.6.34.53:6001 \
STS_SPEECH_LANGUAGES='["en","hi"]' \
STS_TTS_ENGLISH_PROVIDER=indic_parler \
uv run python -m app.server
```

`STS_TTS_ENGLISH_PROVIDER=indic_parler` is optional. It replaces Kokoro for English too, which also removes the GPL espeak dependency.

Measured from this Mac against the US-East server:

| Test | Result |
|---|---|
| Zipformer, Hindi command | Correct (needs 0.8 s of tail padding, otherwise the last word is dropped), ~2 s round trip |
| Zipformer, English command | "please translate this document **in the** English", so Parakeet stays the English engine |
| Parler-TTS | ~1.5× slower than real time (4.7 s to make 3.1 s of audio) |

**The servers are not production-ready.** They have no auth and are open on public IPs. Parler-TTS's documented `/ws/tts` WebSocket stalled after `start`, so the adapter uses `POST /api/synthesize`.

## Configuration

All settings are in [app/config.py](app/config.py) and can be overridden with `STS_*` environment variables or a `.env` file; see `.env.example`.

## Known limitations / next steps

- **Kokoro licensing:** Pipecat's `KokoroTTSService` uses kokoro-onnx, which loads `phonemizer` and `espeak-ng` (both GPL) in-process, for English too. That's fine for development. Before shipping, run Kokoro as a separate service (for example Kokoro-FastAPI) and get legal review.
- **NeMo backend not yet run:** the production Parakeet backend (`STS_STT_BACKEND=nemo`, `uv sync --extra nemo`) is written but untested here because it needs an NVIDIA GPU.
- **Storage:** the document and result store is in memory with a 24 h TTL. Postgres and object storage come later.
- **Not supported yet:** OCR for scanned PDFs, reading tables, DOCX/TXT input, and Hindi/Urdu (see the plan).
- **Wispr:** add a `WisprEngine` next to Parakeet and Zipformer once API access is available.
