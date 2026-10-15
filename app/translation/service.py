"""Chunked document translation with fidelity checks, plus the spoken/text summary."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field

from loguru import logger

from app.config import get_settings
from app.documents.language import language_name
from app.documents.models import Chunk, Document
from app.llm.client import LLMUnavailable, get_llm
from app.llm.prompts import CHUNK_GIST_SYSTEM, GIST_SYSTEM, TRANSLATE_SAME_LANG_SYSTEM, TRANSLATE_SYSTEM
from app.translation.fidelity import check_chunk

ProgressCallback = Callable[[int, int], Awaitable[None]]

# Above this size the gist is built map-reduce style from per-chunk one-liners.
_SINGLE_PASS_GIST_CHARS = 24_000


@dataclass
class TranslationResult:
    request_id: str
    doc_id: str
    filename: str
    source_language: str | None
    target_language: str
    pages: int
    chunks: int
    translated_text: str
    summary_text: str
    warnings: list[str] = field(default_factory=list)
    elapsed_secs: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class TranslationService:
    def __init__(self):
        self._settings = get_settings()
        self._llm = get_llm()

    async def translate_document(
        self, doc: Document, target: str, on_progress: ProgressCallback | None = None
    ) -> TranslationResult:
        started = time.perf_counter()
        source = doc.language or "en"
        same = source == target
        sem = asyncio.Semaphore(self._settings.translate_concurrency)
        total = len(doc.chunks)
        done = 0
        warnings = list(doc.warnings)

        async def run(chunk: Chunk) -> str:
            nonlocal done
            async with sem:
                text, problem = await self._translate_chunk(chunk, source, target, same)
            if problem:
                warnings.append(problem)
            done += 1
            if on_progress:
                await on_progress(done, total)
            return text

        translated = await asyncio.gather(*(run(c) for c in doc.chunks))
        translated_text = "\n\n".join(translated)
        gist = await self._gist(translated_text, translated)
        result = TranslationResult(
            request_id=uuid.uuid4().hex,
            doc_id=doc.doc_id,
            filename=doc.filename,
            source_language=source,
            target_language=target,
            pages=doc.page_count,
            chunks=total,
            translated_text=translated_text,
            summary_text="",
            warnings=warnings,
            elapsed_secs=round(time.perf_counter() - started, 2),
        )
        result.summary_text = self._compose_summary(result, gist)
        logger.info(
            f"translated doc {doc.doc_id}: {total} chunks in {result.elapsed_secs}s, "
            f"{len(warnings) - len(doc.warnings)} chunk warnings"
        )
        return result

    async def _translate_chunk(self, chunk: Chunk, source: str, target: str, same: bool) -> tuple[str, str | None]:
        if same:
            system = TRANSLATE_SAME_LANG_SYSTEM.format(target_name=language_name(target))
        else:
            system = TRANSLATE_SYSTEM.format(source_name=language_name(source), target_name=language_name(target))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": chunk.text}]
        max_tokens = max(256, int(len(chunk.text) / 2.5))  # generous: ~4 chars/token, + headroom

        problems: list[str] = []
        for attempt, temperature in enumerate((0.1, 0.0)):
            try:
                output = await self._llm.complete(messages, temperature=temperature, max_tokens=max_tokens)
            except LLMUnavailable as e:
                problems = [f"LLM error: {e}"]
                break
            problems = check_chunk(
                chunk.text, output, same_language=same, min_chrf=self._settings.same_lang_min_chrf
            )
            if not problems:
                return output, None
            logger.warning(f"chunk {chunk.index} attempt {attempt + 1} failed checks: {problems}")

        pages = f"{chunk.pages[0]}" if len(chunk.pages) == 1 else f"{chunk.pages[0]}-{chunk.pages[-1]}"
        return chunk.text, f"Pages {pages} were kept in the original text ({'; '.join(problems)})."

    async def _gist(self, full_text: str, parts: list[str]) -> str | None:
        max_words = self._settings.summary_max_words - 20  # leave room for the fixed sentence
        try:
            if len(full_text) <= _SINGLE_PASS_GIST_CHARS:
                source = full_text
            else:
                lines = await asyncio.gather(
                    *(
                        self._llm.complete(
                            [{"role": "system", "content": CHUNK_GIST_SYSTEM}, {"role": "user", "content": p[:8000]}],
                            temperature=0.0,
                            max_tokens=60,
                        )
                        for p in parts
                    )
                )
                source = "\n".join(lines)
            return await self._llm.complete(
                [
                    {"role": "system", "content": GIST_SYSTEM.format(max_words=max_words)},
                    {"role": "user", "content": source},
                ],
                temperature=0.0,
                max_tokens=120,
            )
        except LLMUnavailable as e:
            logger.warning(f"gist generation failed: {e}")
            return None

    def _compose_summary(self, r: TranslationResult, gist: str | None) -> str:
        """The factual part is built from metadata, so the spoken claim of what was done
        can never be hallucinated; only the topic sentence comes from the model."""
        parts = [
            f"I've translated your {r.pages}-page document "
            f"from {language_name(r.source_language)} to {language_name(r.target_language)}."
        ]
        if gist:
            parts.append(_limit_words(gist, self._settings.summary_max_words - 20))
        kept = [w for w in r.warnings if w.startswith("Pages ") and "original text" in w]
        if kept:
            parts.append(f"{len(kept)} section{'s' if len(kept) > 1 else ''} could not be verified and were left as written.")
        parts.append("The full text is on your screen.")
        return " ".join(parts)


def _limit_words(text: str, n: int) -> str:
    words = text.split()
    if len(words) <= n:
        return text.strip()
    clipped = " ".join(words[:n]).rstrip(",;:")
    return clipped + ("" if clipped.endswith(".") else ".")
