# STT & TTS Benchmark Plan

**Scope:** English, Hindi, Urdu, Bhojpuri, Awadhi. Self-hostable models only. Quality decides; latency is recorded but isn't a deciding factor for now.

## 0. Start from existing benchmarks

Published results already exist for most candidates: the Open ASR Leaderboard (English), AI4Bharat's Vistaar benchmark (Indic ASR), FLEURS and Kathbath scores in model cards, and published listening scores for the Indic TTS models. We use these to pre-select the candidates below, then test them on a small sample of our own material.

- **If they perform well enough** on our cases, we shortlist straight from these results.
- **If not, or if a language has no published results** (Bhojpuri and Awadhi in particular), we run the full benchmark in steps 1–5.

## 1. Build the test set (fixed, reused for every model)

- **STT:** for each language, about 100 utterances, mixing our real commands with general sentences and including some Hinglish. Sources: our own recordings from target speakers, plus Vaani and Rural_Women_Bhojpuri for Bhojpuri and Awadhi. Each utterance has a reference transcript checked by a native speaker.
- **TTS:** for each language, about 30 sentences taken from real summaries, including numbers, dates, currency and English terms ("Rs 25,000", "11:00 IST", "VPN").

## 2. Gate (pass/fail before any testing)

- Self-hostable on our GPU.
- Officially supports the language.

## 3. Run every candidate the same way

- Same L40S, same audio and texts, and each model wired in through our provider registry (a small adapter per model).
- One script produces one results table.

## 4. Measure

| | STT | TTS |
|---|---|---|
| **Quality (decides)** | WER and CER per language; command accuracy (intent + language right); Hindi/Urdu confusion | Native-speaker rating 1–5 (3+ raters per language); intelligibility (run our STT on the output); numbers and terms read correctly |
| **Recorded only** | Latency p50/p95, GPU memory | Time to first audio, real-time factor, GPU memory |

## 5. Shortlist

- For each language, rank the candidates by quality.
- Shortlist the top 2 that pass a minimum bar (for example, command accuracy ≥ 95% for STT, an average rating ≥ 3.5 for TTS).
- Where nothing passes (likely Bhojpuri/Awadhi TTS), mark it as a gap and plan a fine-tune: for example, Indic Parler-TTS on Bhojpuri/Awadhi audio from Vaani.

## Starting candidates

Pre-selected from published results (step 0). Lower WER/CER is better. "VoI" is the Voice of India benchmark of phone-call speech, and its numbers are self-reported by Bodhan AI.

### STT

| Model | Languages | Published results | License |
|---|---|---|---|
| Indic-Transcribe-core / -flex (Bodhan AI + AI4Bharat, 1.2B) | en (Indian accent), hi, ur, bho | VoI WER, core: hi 3.5, ur 5.0, bho 13.3. Flex adds Hinglish output | Indic Open Model License (gated) |
| Omnilingual ASR (Meta, LLM-7B) | en, hi, ur, bho, **awa** | CER: hi 3.6, ur 3.5, bho 16.5, awa 2.2, en 3.0 | Apache-2.0 |
| IndicConformer-600M (AI4Bharat) | hi, ur | VoI WER: hi 6.5, ur 8.0 | MIT (gated) |
| Nemotron-3.5-ASR-Streaming-0.6B (NVIDIA) | en, hi | FLEURS WER: hi 6.8–8.1, en 7.9–9.4 (streaming) | OpenMDW-1.1 |
| Parakeet-TDT-0.6B-v2 / Canary-Qwen-2.5B (NVIDIA) | en | Open ASR Leaderboard WER: 6.05 / 5.63 | CC-BY-4.0 |
| **Gemma 4 audio input** (E4B / 12B) | en, hi, ur, bho (not officially listed) | VoI WER (E4B): hi 9.1, ur 14.2, bho 27.0. Vaani Hindi (12B): 24.6 | Apache-2.0 |
| Indic Zipformer (team server) | en, hi + 10 Indic | Our test: Hindi command correct | Team-hosted |

**Gemma 4 as STT:** it could replace the STT step entirely, since one model would hear the command and act on it. Its Indic accuracy trails the specialist models (Hindi 9.1 vs 3.5). Runtime support also matters:
- Audio input works in Hugging Face Transformers for 12B.
- vLLM supports audio only for E2B and E4B.
- Ollama lists gemma4:12b as text and image only. Its audio support was added for the Mac (MLX) engine, so it must be checked on our EC2 setup.

We test it alongside the specialist models.

### TTS

| Model | Languages | Published results | License |
|---|---|---|---|
| Indic-Speak (Bodhan AI, 3.8B) | en, hi, ur | MOS 4.46; 95 voices | Indic Open Model License (approval needed to serve as an API) |
| Indic Parler-TTS (AI4Bharat, 0.9B) | en (Indian accent), hi, ur | MOS 4.14; 69 voices; fine-tunable | Apache-2.0 |
| Veena (Maya Research, ~3B) | en, hi, Hinglish | MOS 4.2, 98% intelligibility (self-reported); no streaming yet | Apache-2.0 |
| Maya1 (Maya Research, 3.3B) | Hindi not stated | None published | Apache-2.0 |
| IndicF5 (AI4Bharat, 0.4B) | hi | None published; voice cloning | MIT (gated) |
| Chatterbox Multilingual (0.5B) | en, hi | No metrics | MIT |
| Kokoro-82M | en, hi | Hindi voices rated "C" by Kokoro's own authors; baseline only | Apache-2.0 |

### Bhojpuri / Awadhi TTS

No open model exists for either, apart from Meta MMS-TTS Awadhi, which is non-commercial and basic quality. Plan:
- Fine-tune Indic Parler-TTS on cleaned Vaani / SRUTI Bhojpuri audio, plus a few hours of commissioned recordings for Awadhi.
- Stopgap: read the Devanagari text with a Hindi voice.

No published benchmark tests how numbers or English terms inside Hindi speech are read, so step 1 covers it.
