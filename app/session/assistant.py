"""The session "brain": a Pipecat FrameProcessor between turn detection and TTS.

It collects the user's transcript for a turn, gates on the detected spoken language,
parses the command, runs the translation job in the background, and emits two outputs:
- text to the client as RTVI ``server-message`` frames (over the WebRTC data channel)
- speech, by pushing ``TTSSpeakFrame`` downstream to Kokoro.
"""

import asyncio
import re
import time
from collections.abc import Callable

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    ManuallySwitchServiceFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from app.commands.parser import Command, Intent, parse_command
from app.config import get_settings
from app.documents.language import language_name
from app.documents.models import Document, DocumentStatus
from app.lid.speech_lid import LanguageDetectedFrame
from app.translation.service import TranslationService

_MAX_CLARIFICATIONS = 2
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


class DocumentAssistant(FrameProcessor):
    def __init__(
        self,
        *,
        session_id: str,
        get_document: Callable[[str], Document | None],
        save_result: Callable[[dict], None],
        tts_for_language: Callable[[str], FrameProcessor | None] = lambda lang: None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._settings = get_settings()
        self._session_id = session_id
        self._get_document = get_document
        self._save_result = save_result
        self._tts_for_language = tts_for_language
        self._active_tts: FrameProcessor | None = None
        self._translator = TranslationService()

        self._turn_text: list[str] = []
        self._turn_languages: list[LanguageDetectedFrame] = []
        self._turn_ended = False
        self._clarifications = 0
        self._job: asyncio.Task | None = None
        self._last_summary: str | None = None

    # ------------------------------------------------------------------ frames

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LanguageDetectedFrame):
            self._turn_languages.append(frame)
            await self._send("language", frame.language, confidence=round(frame.confidence, 3), top=frame.top)
            return  # consumed
        if isinstance(frame, TranscriptionFrame):
            self._turn_text.append(frame.text)
            if self._turn_ended:  # transcript arrived after the turn already ended
                await self._finish_turn()
            return  # consumed: never forward user text to TTS
        if isinstance(frame, UserStartedSpeakingFrame):
            self._turn_ended = False
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_ended = True
            if self._turn_text:
                await self._finish_turn()

        await self.push_frame(frame, direction)

    async def _finish_turn(self):
        text = " ".join(t.strip() for t in self._turn_text if t.strip())
        languages = self._turn_languages
        self._turn_text, self._turn_languages, self._turn_ended = [], [], False
        if text:
            # Command handling may call the LLM; never block the frame loop on it.
            self.create_task(self._handle_turn(text, languages), name="handle_turn")

    # ------------------------------------------------------------------ turn logic

    async def _handle_turn(self, text: str, languages: list[LanguageDetectedFrame]):
        started = time.perf_counter()
        if rejected := self._rejected_language(languages):
            name = language_name(rejected)
            await self._reply(
                f"I heard {name}. Right now I only understand English. Please say your command in English.",
                kind="language_rejected",
                detected=rejected,
            )
            return

        command = await parse_command(text)
        logger.info(f"command {command} from {text!r} in {time.perf_counter() - started:.2f}s")
        await self._send("command", command.intent.value, target_language=command.target_language, heard=text)

        match command.intent:
            case Intent.STOP:
                await self._stop_job()
            case Intent.REPEAT_SUMMARY:
                await self._reply(self._last_summary or "There is nothing to repeat yet.", kind="repeat")
            case Intent.HELP:
                await self._reply(
                    "Upload a PDF, then say something like: translate this document into English.", kind="help"
                )
            case Intent.TRANSLATE:
                await self._handle_translate(command)
            case _:
                await self._clarify(
                    "Sorry, I didn't catch a command. You can say: translate this document into English."
                )

    def _rejected_language(self, languages: list[LanguageDetectedFrame]) -> str | None:
        """Reject only when LID is confident the turn is in an unsupported language.

        Low-confidence or too-short segments are accepted: false rejections of accented
        English are worse than letting the English-only STT try.
        """
        judged = [f for f in languages if f.language]
        if not judged:
            return None
        best = max(judged, key=lambda f: f.duration_secs)
        if best.language in self._settings.stt:  # a language with an STT engine
            return None
        if best.confidence >= self._settings.lid_reject_confidence:
            return best.language
        return None

    async def _handle_translate(self, command: Command):
        supported = self._settings.supported_languages
        target = command.target_language
        if not target:
            if self._clarifications >= _MAX_CLARIFICATIONS:
                target = supported[0]  # stop asking; use the only/primary option and say so
            else:
                await self._clarify("Which language should I translate the document into? Right now I support English.")
                return
        if target not in supported:
            await self._reply(
                f"I can't translate into {language_name(target)} yet. Only English output is available right now.",
                kind="target_unsupported",
            )
            return
        self._clarifications = 0

        doc = self._get_document(self._session_id)
        if doc is None:
            await self._reply("Please upload a PDF first, then ask me again.", kind="no_document")
            return
        if doc.status == DocumentStatus.PROCESSING:
            await self._reply("Your PDF is still being processed. Please ask again in a moment.", kind="doc_processing")
            return
        if doc.status == DocumentStatus.REJECTED:
            await self._reply(f"I couldn't use that PDF. {doc.error}", kind="doc_rejected")
            return
        if self._job and not self._job.done():
            await self._reply("I'm already translating. Say stop to cancel.", kind="busy")
            return

        estimate = max(5, int(doc.char_count / 1000 * 1.5))  # rough, tuned on the pilot GPU
        await self._reply(
            f"Translating your {doc.page_count}-page document into {language_name(target)}. "
            f"This may take about {estimate} seconds.",
            kind="ack",
        )
        self._job = self.create_task(self._run_job(doc, target), name="translation_job")

    async def _run_job(self, doc: Document, target: str):
        async def on_progress(done: int, total: int):
            await self._send("progress", None, done=done, total=total)

        try:
            result = await self._translator.translate_document(doc, target, on_progress)
        except asyncio.CancelledError:
            await self._send("cancelled", None)
            raise
        except Exception as e:
            logger.exception("translation job failed")
            await self._reply("Sorry, the translation failed. Please try again.", kind="error", error=str(e))
            return

        payload = result.to_dict()
        self._save_result(payload)
        self._last_summary = result.summary_text
        # Text output: summary + metadata inline; the full text is fetched over HTTP
        # because data channel messages have size limits.
        meta = {k: v for k, v in payload.items() if k != "translated_text"}
        preview = result.translated_text[:20_000]
        await self._send(
            "result",
            None,
            **meta,
            translated_preview=preview,
            truncated=len(preview) < len(result.translated_text),
            result_url=f"/api/results/{result.request_id}",
        )
        await self._speak(result.summary_text)

    async def _stop_job(self):
        if self._job and not self._job.done():
            await self.cancel_task(self._job)
            self._job = None
            await self._reply("Stopped.", kind="stopped")
        else:
            await self._reply("There's nothing to stop.", kind="stopped")

    async def _clarify(self, question: str):
        self._clarifications += 1
        await self._reply(question, kind="clarify")

    # ------------------------------------------------------------------ outputs

    async def _reply(self, text: str, *, kind: str, **extra):
        await self._send("assistant", text, kind=kind, **extra)
        await self._speak(text)

    async def _speak(self, text: str, language: str = "en"):
        """Speak ``text`` (written in ``language``) with the TTS service for that language."""
        service = self._tts_for_language(language)
        if service is not None and service is not self._active_tts:
            await self.push_frame(ManuallySwitchServiceFrame(service=service))
            self._active_tts = service
        # One TTS request per sentence: first audio starts sooner, and no single
        # request runs long enough to hit the TTS service's no-audio timeout.
        for sentence in _SENTENCE_END_RE.split(text.strip()):
            if sentence:
                await self.push_frame(TTSSpeakFrame(sentence))

    async def _send(self, type_: str, text: str | None, **data):
        await self.push_frame(RTVIServerMessageFrame(data={"type": type_, "text": text, **data}))
