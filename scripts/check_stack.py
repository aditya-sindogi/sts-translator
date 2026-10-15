"""Check that every component works on this machine, and print PASS/FAIL per check.

    uv run python -m scripts.check_stack                # full check (LLM must be running)
    uv run python -m scripts.check_stack --models-only  # just download/load the local models

Checks: LLM endpoint (model served, thinking off, JSON output), Parakeet STT on a spoken
sample, spoken language ID, Kokoro TTS, and the optional Indic ASR/TTS servers.
"""

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time

import numpy as np

from app.config import get_settings

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def speech_16k(text: str, voice: str = "am_michael", lang: str = "en-us") -> bytes:
    from scripts.voice_e2e import synthesize

    return synthesize(text, voice, lang)[::3].astype(np.int16).tobytes()  # 48 kHz -> 16 kHz


async def check_llm():
    from app.commands.parser import parse_command
    from app.llm.client import get_llm

    s = get_settings()
    llm = get_llm()
    try:
        models = await llm._client.models.list()
        served = [m.id for m in models.data]
        record("LLM model served", s.llm_model in served, f"{s.llm_base_url} serves {served}")
    except Exception as e:
        record("LLM model served", False, f"{s.llm_base_url}: {e}")
        return

    started = time.perf_counter()
    resp = await llm._client.chat.completions.create(
        model=s.llm_model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=50,
        temperature=0,
        extra_body=s.llm_extra_body,
    )
    tokens = resp.usage.completion_tokens if resp.usage else -1
    text = (resp.choices[0].message.content or "").strip()
    record(
        "LLM thinking off",
        0 <= tokens < 20,
        f"{text!r}, {tokens} completion tokens in {time.perf_counter() - started:.2f}s",
    )

    cmd = await parse_command("could you please put this document into english for me")
    record("LLM command parsing (JSON)", cmd.intent.value == "translate" and cmd.target_language == "en", str(cmd))


async def check_stt_lid(pcm: bytes):
    from app.lid.speech_lid import get_identifier
    from app.stt.engines import ParakeetEngine

    started = time.perf_counter()
    engine = await asyncio.to_thread(ParakeetEngine)
    record("Parakeet loaded", True, f"{engine.name} in {time.perf_counter() - started:.1f}s")
    await engine.transcribe(pcm, "en")  # warm-up
    started = time.perf_counter()
    text = await engine.transcribe(pcm, "en")
    ok = "translate" in text.lower() and "english" in text.lower()
    record("Parakeet transcription", ok, f"{text!r} in {time.perf_counter() - started:.2f}s")

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    lang, conf, _ = await asyncio.to_thread(get_identifier().classify, audio)
    record("Spoken language ID", lang == "en", f"{lang} ({conf:.2f})")


async def check_indic(pcm: bytes):
    s = get_settings()
    if s.indic_asr_url:
        from app.stt.indic_zipformer import IndicZipformerEngine

        try:
            text = await IndicZipformerEngine(s.indic_asr_url, ["en"]).transcribe(pcm, "en")
            record("Indic ASR server", bool(text), f"{s.indic_asr_url}: {text!r}")
        except Exception as e:
            record("Indic ASR server", False, f"{s.indic_asr_url}: {e}")
    if s.indic_tts_url:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10) as c:
                health = (await c.get(f"{s.indic_tts_url.rstrip('/')}/health")).json()
            record("Indic TTS server", health.get("status") == "ok", f"{s.indic_tts_url}: {health}")
        except Exception as e:
            record("Indic TTS server", False, f"{s.indic_tts_url}: {e}")


def check_gpu():
    if not shutil.which("nvidia-smi"):
        return
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"INFO  GPU: {out}")


async def main(models_only: bool):
    s = get_settings()
    print(f"INFO  LLM={s.llm_base_url} model={s.llm_model} stt_backend={s.stt_backend}")

    started = time.perf_counter()
    pcm = await asyncio.to_thread(speech_16k, "Please translate this document into English.")
    record("Kokoro TTS", len(pcm) > 16000, f"{len(pcm) / 32000:.1f}s of speech in {time.perf_counter() - started:.1f}s")

    if models_only:
        from pipecat.services.moonshine.stt import MoonshineSTTService

        await check_stt_lid(pcm)
        MoonshineSTTService()  # downloads the CPU fallback model
        record("Moonshine fallback", True, "downloaded")
    else:
        await check_llm()
        await check_stt_lid(pcm)
        await check_indic(pcm)
    check_gpu()

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + ("ALL CHECKS PASSED" if not failed else f"FAILED: {', '.join(failed)}"))
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-only", action="store_true", help="download/load local models, skip the LLM")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.models_only)))
