"""The session "brain": a Pipecat FrameProcessor between turn detection and TTS.

It collects the user's transcript for a turn, gates on the detected spoken language,
parses the command, runs the translation job in the background, and emits two outputs:
- text to the client as RTVI ``server-message`` frames (over the WebRTC data channel)
- speech, by pushing ``TTSSpeakFrame`` downstream to the TTS service.

It replies in the language the user spoke (from the STT transcript), when that language
has messages (app/session/messages.py) and a voice in STS_TTS; otherwise in English.
"""

import asyncio
import re
import time
from collections.abc import Callable

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
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
from app.documents.models import Document, DocumentStatus
from app.lid.speech_lid import LanguageDetectedFrame
from app.session.messages import MESSAGES, name_of, options, say
from app.session.mic_gate import MicGate
from app.translation.service import TranslationService

_MAX_CLARIFICATIONS = 2
# A turn that isn't a clear command may be the first half of one ("this document..." pause
# "...into Hindi"). Wait this long before saying "I didn't understand", and join it with
# speech that follows within the merge window.
_UNCLEAR_REPLY_DELAY_SECS = 2.5
_FRAGMENT_MERGE_SECS = 6.0
# Reopen the mic this long after the bot goes quiet (bridges pauses between sentences),
# and never stay deaf longer than the watchdog if speech never starts (e.g. TTS failed).
_REOPEN_AFTER_SPEECH_SECS = 2.0
_MIC_WATCHDOG_SECS = 30.0
# Sentence ends: Latin . ! ?, Devanagari danda । and Urdu full stop ۔ / question mark ؟
_SENTENCE_END_RE = re.compile(r"(?<=[.!?\u0964\u06d4\u061f])\s+")


class DocumentAssistant(FrameProcessor):
    def __init__(
        self,
        *,
        session_id: str,
        get_document: Callable[[str], Document | None],
        save_result: Callable[[dict], None],
        tts_for_language: Callable[[str], FrameProcessor | None] = lambda lang: None,
        mic_gate: MicGate | None = None,
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
        self._turn_spoken: list[str] = []  # language of each transcript in this turn
        self._turn_languages: list[LanguageDetectedFrame] = []
        self._turn_ended = False
        self._clarifications = 0
        self._job: asyncio.Task | None = None
        self._last_summary: tuple[str, str] | None = None  # (text, language)
        self._lang = "en"  # reply language for the current turn
        self._fragment: tuple[str, float] | None = None  # (unclear text, time) awaiting more speech
        self._unclear_reply: asyncio.Task | None = None
        self._mic = mic_gate or MicGate()
        self._reopen: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None

    # ------------------------------------------------------------------ frames

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (LanguageDetectedFrame, TranscriptionFrame)) and self._mic.closed:
            # Speech that was already in the pipeline when we got busy: ignore it too.
            logger.debug(f"ignored while busy: {getattr(frame, 'text', None) or frame.language}")
            return
        if isinstance(frame, LanguageDetectedFrame):
            self._turn_languages.append(frame)
            await self._send("language", frame.language, confidence=round(frame.confidence, 3), top=frame.top)
            return  # consumed
        if isinstance(frame, TranscriptionFrame):
            self._turn_text.append(frame.text)
            if frame.language:
                self._turn_spoken.append(str(frame.language.value if hasattr(frame.language, "value") else frame.language))
            if self._turn_ended:  # transcript arrived after the turn already ended
                await self._finish_turn()
            return  # consumed: never forward user text to TTS
        if isinstance(frame, BotStartedSpeakingFrame):
            await self._cancel(self._reopen)
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._mic.closed and not self._job_running():
            await self._cancel(self._reopen)
            self._reopen = self.create_task(self._reopen_mic_later(), name="reopen_mic")
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._turn_ended = False
            await self._cancel_unclear_reply()  # they're still talking: hold the "didn't understand"
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_ended = not self._mic.closed
            if self._turn_text and not self._mic.closed:
                await self._finish_turn()

        await self.push_frame(frame, direction)

    async def _finish_turn(self):
        text = " ".join(t.strip() for t in self._turn_text if t.strip())
        languages, spoken = self._turn_languages, self._turn_spoken
        if text:
            await self._close_mic()  # busy from here until the answer has been spoken
        self._turn_text, self._turn_languages, self._turn_spoken, self._turn_ended = [], [], [], False
        if text:
            # Command handling may call the LLM; never block the frame loop on it.
            self.create_task(self._handle_turn(text, languages, spoken), name="handle_turn")

    # ------------------------------------------------------------------ turn logic

    async def _handle_turn(self, text: str, languages: list[LanguageDetectedFrame], spoken: list[str]):
        started = time.perf_counter()
        if rejected := self._rejected_language(languages):
            self._lang = "en"  # we can't speak an unsupported language, so explain in English
            understood = options(list(self._settings.stt), "en")
            await self._reply(
                f"I heard {name_of(rejected, 'en')}. Right now I understand {understood}. "
                "Please say your command in one of those.",
                kind="language_rejected",
                detected=rejected,
            )
            return
        self._lang = reply_language(spoken[-1] if spoken else "en", self._settings.tts)

        text = self._merge_fragment(text)
        command = await parse_command(text)
        logger.info(f"command {command} from {text!r} in {time.perf_counter() - started:.2f}s")
        await self._send("command", command.intent.value, target_language=command.target_language, heard=text)

        match command.intent:
            case Intent.STOP:
                await self._stop_job()
            case Intent.REPEAT_SUMMARY:
                if self._last_summary:
                    summary, language = self._last_summary
                    await self._send("assistant", summary, kind="repeat")
                    await self._speak(summary, language)
                else:
                    await self._say("nothing_to_repeat", kind="repeat")
            case Intent.HELP:
                await self._say("help", kind="help")
            case Intent.TRANSLATE:
                await self._handle_translate(command)
            case _:
                self._hold_fragment(text)

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
        choices = options(supported, self._lang)
        target = command.target_language
        if not target:
            if self._clarifications >= _MAX_CLARIFICATIONS:
                target = supported[0]  # stop asking; use the primary option
            else:
                self._clarifications += 1
                await self._say("ask_target", kind="clarify", options=choices)
                return
        if target not in supported:
            await self._say(
                "target_unsupported", kind="target_unsupported", target=name_of(target, self._lang), options=choices
            )
            return
        self._clarifications = 0

        doc = self._get_document(self._session_id)
        if doc is None:
            await self._say("no_document", kind="no_document")
            return
        if doc.status == DocumentStatus.PROCESSING:
            await self._say("doc_processing", kind="doc_processing")
            return
        if doc.status == DocumentStatus.REJECTED:
            await self._say("doc_rejected", kind="doc_rejected", reason=doc.error or "")
            return
        if self._job and not self._job.done():
            await self._say("busy", kind="busy")
            return

        estimate = max(5, int(doc.char_count / 1000 * 1.5))  # rough, tuned on the pilot GPU
        await self._say("ack", kind="ack", pages=doc.page_count, target=name_of(target, self._lang), secs=estimate)
        self._job = self.create_task(self._run_job(doc, target, self._lang), name="translation_job")

    async def _run_job(self, doc: Document, target: str, reply_lang: str):
        async def on_progress(done: int, total: int):
            await self._send("progress", None, done=done, total=total)

        try:
            result = await self._translator.translate_document(doc, target, on_progress, summary_language=reply_lang)
        except asyncio.CancelledError:
            await self._send("cancelled", None)
            raise
        except Exception as e:
            logger.exception("translation job failed")
            await self._say("error", kind="error", language=reply_lang, error=str(e))
            await self._close_mic()
            return

        payload = result.to_dict()
        self._save_result(payload)
        self._last_summary = (result.summary_text, reply_lang)
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
        await self._speak(result.summary_text, reply_lang)
        await self._close_mic()  # reopens after the summary has been spoken

    async def _stop_job(self):
        if self._job and not self._job.done():
            await self.cancel_task(self._job)
            self._job = None
            await self._say("stopped", kind="stopped")
        else:
            await self._say("nothing_to_stop", kind="stopped")

    # ------------------------------------------------------------------ paused commands

    def _merge_fragment(self, text: str) -> str:
        """Prepend a recent unclear fragment, so a command split by a pause is read whole."""
        fragment, self._fragment = self._fragment, None
        if fragment and time.monotonic() - fragment[1] <= _FRAGMENT_MERGE_SECS:
            return f"{fragment[0]} {text}"
        return text

    def _hold_fragment(self, text: str):
        """Keep an unclear turn for joining, and only say "I didn't understand" if nothing follows."""
        self._fragment = (text, time.monotonic())
        self._unclear_reply = self.create_task(self._reply_unclear_later(), name="unclear_reply")
        self._mic.closed = False  # keep listening for the rest of the command
        self._send_mic_state()

    async def _reply_unclear_later(self):
        await asyncio.sleep(_UNCLEAR_REPLY_DELAY_SECS)
        self._unclear_reply = None
        self._clarifications += 1
        await self._say("unclear", kind="clarify")

    # ------------------------------------------------------------------ mic gate

    def _job_running(self) -> bool:
        return self._job is not None and not self._job.done()

    async def _close_mic(self):
        if not self._mic.closed:
            self._mic.closed = True
            self._send_mic_state()
        await self._cancel(self._watchdog)
        self._watchdog = self.create_task(self._mic_watchdog(), name="mic_watchdog")

    def _open_mic(self):
        if self._mic.closed:
            self._mic.closed = False
            self._send_mic_state()

    async def _reopen_mic_later(self):
        await asyncio.sleep(_REOPEN_AFTER_SPEECH_SECS)
        self._reopen = None
        if not self._job_running():
            self._open_mic()

    async def _mic_watchdog(self):
        """Never stay deaf forever: reopen if the bot never spoke (or a job hung)."""
        while True:
            await asyncio.sleep(_MIC_WATCHDOG_SECS)
            if not self._job_running():
                self._watchdog = None
                self._open_mic()
                return

    def _send_mic_state(self):
        self.create_task(self._send("mic", None, listening=not self._mic.closed), name="mic_state")

    async def _cancel(self, task: asyncio.Task | None):
        if task and not task.done():
            await self.cancel_task(task)

    async def _cancel_unclear_reply(self):
        if self._unclear_reply and not self._unclear_reply.done():
            await self.cancel_task(self._unclear_reply)
        self._unclear_reply = None

    # ------------------------------------------------------------------ outputs

    async def _say(self, key: str, *, kind: str, language: str | None = None, error: str | None = None, **values):
        """Say message ``key`` in the reply language (text to the page + speech)."""
        language = language or self._lang
        extra = {"error": error} if error else {}
        await self._reply(say(key, language, **values), kind=kind, language=language, **extra)

    async def _reply(self, text: str, *, kind: str, language: str | None = None, **extra):
        language = language or self._lang
        await self._send("assistant", text, kind=kind, language=language, **extra)
        await self._speak(text, language)

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


def reply_language(spoken: str, tts: dict[str, str]) -> str:
    """Reply in the spoken language if we have messages and a voice for it, else English."""
    return spoken if spoken in MESSAGES and spoken in tts else "en"
