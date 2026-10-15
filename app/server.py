"""HTTP + WebRTC signalling server.

    POST /api/documents          upload a PDF for a session -> {doc_id, status}
    GET  /api/documents/{doc_id} parse status, page count, warnings
    GET  /api/results/{id}       full translation result (JSON) or ?format=txt
    POST /api/offer              WebRTC SDP offer -> answer; starts a Pipecat pipeline
    GET  /api/config             client settings (ICE servers)
    GET  /                       test client

Run:  uv run python -m app.server
"""

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from loguru import logger

from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequest, SmallWebRTCRequestHandler

from app.bot import run_bot
from app.config import ROOT, get_settings
from app.documents.store import get_store
from app.lid.speech_lid import get_identifier
from app.llm.client import LLMUnavailable, get_llm
from app.stt.parakeet import get_backend

_sessions: set[asyncio.Task] = set()


def _prefetch_session_models():
    """Download per-session model files now, so the first caller doesn't wait for them."""
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files
    from pipecat.services.moonshine.stt import MoonshineSTTService

    _ensure_model_files(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", KOKORO_CACHE_DIR / "voices-v1.0.bin")
    if get_settings().stt_moonshine_fallback:
        MoonshineSTTService()  # constructing it downloads and caches the model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load shared models once, before the first call, so the first user isn't the warm-up.
    logger.info("warming up models (first run downloads them)…")
    await asyncio.gather(
        asyncio.to_thread(get_identifier),
        asyncio.to_thread(get_backend),
        asyncio.to_thread(_prefetch_session_models),
    )
    s = get_settings()
    logger.info(f"LLM endpoint: {s.llm_base_url} model={s.llm_model}")
    try:
        await get_llm().complete([{"role": "user", "content": "Say OK."}], max_tokens=5)
        logger.info("LLM ready")
    except LLMUnavailable as e:
        logger.error(f"LLM not reachable at {s.llm_base_url}: {e}")
        if s.llm_required:
            raise RuntimeError("LLM is required (STS_LLM_REQUIRED=true) but unreachable") from e
    yield
    for task in list(_sessions):
        task.cancel()


app = FastAPI(title="sts-up", lifespan=lifespan)
webrtc = SmallWebRTCRequestHandler(ice_servers=get_settings().ice_servers or None)


@app.get("/")
async def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/config")
async def client_config():
    """Settings the browser needs (ICE servers for WebRTC)."""
    return {"ice_servers": [{"urls": url} for url in get_settings().ice_servers]}


@app.post("/api/documents")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    settings = get_settings()
    data = await file.read(settings.max_pdf_mb * 1024 * 1024 + 1)
    doc = await get_store().ingest(session_id=session_id, filename=file.filename or "document.pdf", data=data)
    return doc.summary()


@app.get("/api/documents/{doc_id}")
async def document_status(doc_id: str):
    doc = get_store().get(doc_id)
    if not doc:
        raise HTTPException(404, "document not found or expired")
    return doc.summary()


@app.get("/api/results/{request_id}")
async def get_result(request_id: str, format: str = "json"):
    result = get_store().get_result(request_id)
    if not result:
        raise HTTPException(404, "result not found or expired")
    if format == "txt":
        return PlainTextResponse(result["translated_text"])
    return result


@app.post("/api/offer")
async def offer(request: Request):
    body = await request.json()
    session_id = (body.get("request_data") or body.get("requestData") or {}).get("session_id")
    if not session_id:
        raise HTTPException(400, "request_data.session_id is required")

    async def on_connection(connection):
        task = asyncio.create_task(run_bot(connection, session_id))
        _sessions.add(task)
        task.add_done_callback(_sessions.discard)

    return await webrtc.handle_web_request(SmallWebRTCRequest.from_dict(body), on_connection)


def main():
    s = get_settings()
    uvicorn.run(app, host=s.host, port=s.port)


if __name__ == "__main__":
    main()
