"""Headless end-to-end voice test over real WebRTC (no browser needed).

Synthesizes a spoken command with Kokoro, uploads a PDF, connects to the running server
exactly like the web client (SmallWebRTC + RTVI client-ready + pings), plays the command
into the mic track, and prints every event the bot sends back until the result arrives.

    uv run python -m app.server            # in another terminal
    uv run python -m scripts.voice_e2e samples/sample.pdf "Please translate this document into English."
    uv run python -m scripts.voice_e2e samples/sample.pdf "कृपया इस दस्तावेज़ का अनुवाद करें" hf_alpha hi
"""

import asyncio
import fractions
import json
import sys
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame

SERVER = "http://127.0.0.1:7860"
RATE = 48000
FRAME = 960  # 20 ms


def synthesize(text: str, voice: str = "am_michael", lang: str = "en-us") -> np.ndarray:
    """Kokoro speech as int16 mono at 48 kHz."""
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files

    model, voices = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", KOKORO_CACHE_DIR / "voices-v1.0.bin"
    _ensure_model_files(model, voices)
    samples, sr = Kokoro(str(model), str(voices)).create(text, voice=voice, speed=1.0, lang=lang)
    n = int(len(samples) * RATE / sr)
    resampled = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples)
    return (np.clip(resampled, -1, 1) * 32767).astype(np.int16)


class ScriptedMic(MediaStreamTrack):
    """1 s silence, the utterance, then silence forever (paced in real time)."""

    kind = "audio"

    def __init__(self, speech: np.ndarray):
        super().__init__()
        self._audio = np.concatenate([np.zeros(RATE, np.int16), speech])
        self._pos = 0
        self._start = None

    async def recv(self) -> AudioFrame:
        if self._start is None:
            self._start = time.monotonic()
        target = self._start + self._pos / RATE
        await asyncio.sleep(max(0, target - time.monotonic()))
        chunk = self._audio[self._pos : self._pos + FRAME]
        if len(chunk) < FRAME:
            chunk = np.concatenate([chunk, np.zeros(FRAME - len(chunk), np.int16)])
        frame = AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = RATE, self._pos, fractions.Fraction(1, RATE)
        self._pos += FRAME
        return frame


async def main(pdf_path: str, command: str, voice: str = "am_michael", lang: str = "en-us", timeout: float = 180):
    session_id = str(uuid.uuid4())
    t0 = time.monotonic()
    stamp = lambda: f"[{time.monotonic() - t0:6.2f}s]"  # noqa: E731

    async with httpx.AsyncClient(base_url=SERVER, timeout=30) as http:
        if pdf_path != "-":
            files = {"file": (Path(pdf_path).name, Path(pdf_path).read_bytes(), "application/pdf")}
            doc = (await http.post("/api/documents", data={"session_id": session_id}, files=files)).json()
            while doc["status"] == "processing":
                await asyncio.sleep(0.2)
                doc = (await http.get(f"/api/documents/{doc['doc_id']}")).json()
            print(stamp(), "document:", {k: doc[k] for k in ("status", "pages", "chunks", "language", "error")})

        print(stamp(), f"synthesizing command: {command!r}")
        speech = synthesize(command, voice, lang)

        pc = RTCPeerConnection()
        pc.addTrack(ScriptedMic(speech))
        pc.addTransceiver("video", direction="sendrecv")
        done = asyncio.Event()
        bot_audio = {"frames": 0, "voiced": 0}

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":

                async def drain():
                    while True:
                        try:
                            f = await track.recv()
                        except Exception:
                            return
                        bot_audio["frames"] += 1
                        if np.abs(f.to_ndarray()).max() > 500:
                            bot_audio["voiced"] += 1

                asyncio.ensure_future(drain())

        dc = pc.createDataChannel("chat", ordered=True)

        @dc.on("open")
        def on_open():
            dc.send(json.dumps({"id": "1", "label": "rtvi-ai", "type": "client-ready",
                                "data": {"version": "2.1.0", "about": {"library": "voice_e2e"}}}))

            async def ping():
                while dc.readyState == "open":
                    dc.send(f"ping: {time.time()}")
                    await asyncio.sleep(1)

            asyncio.ensure_future(ping())

        @dc.on("message")
        def on_message(raw):
            m = json.loads(raw)
            if m.get("label") != "rtvi-ai":
                return
            t, d = m.get("type"), m.get("data") or {}
            if t == "user-transcription" and d.get("final"):
                print(stamp(), "YOU   :", d.get("text"))
            elif t == "server-message":
                kind = d.get("type")
                if kind == "result":
                    print(stamp(), "RESULT:", {k: d[k] for k in ("pages", "source_language", "target_language", "chunks", "elapsed_secs", "warnings")})
                    print(stamp(), "SPOKEN:", d["summary_text"])
                    print(stamp(), "TEXT  :", d["translated_preview"][:300].replace("\n", " ") + " …")
                    asyncio.get_event_loop().call_later(12, done.set)  # let the summary audio play
                elif kind == "assistant":
                    print(stamp(), "BOT   :", d.get("text"))
                    if d.get("kind") not in ("ack",):
                        asyncio.get_event_loop().call_later(8, done.set)
                else:
                    print(stamp(), f"{kind:6}:", {k: v for k, v in d.items() if k != "type"})
            elif t in ("bot-started-speaking", "bot-stopped-speaking", "error"):
                print(stamp(), t, d or "")

        await pc.setLocalDescription(await pc.createOffer())
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)
        answer = (await http.post("/api/offer", json={
            "sdp": pc.localDescription.sdp, "type": pc.localDescription.type,
            "request_data": {"session_id": session_id}})).json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
        print(stamp(), "connected; speaking…")

        try:
            await asyncio.wait_for(done.wait(), timeout)
        except asyncio.TimeoutError:
            print(stamp(), "TIMEOUT waiting for the bot")
        voiced_secs = bot_audio["voiced"] * 0.02
        print(stamp(), f"bot audio received: {voiced_secs:.1f}s of speech")
        await pc.close()


if __name__ == "__main__":
    # Optional: voice and language, e.g. hf_alpha hi  (Kokoro Hindi voice) to test LID rejection.
    args = sys.argv[1:] + [None] * 4
    asyncio.run(main(args[0], args[1] or "Please translate this document into English.", args[2] or "am_michael", args[3] or "en-us"))
