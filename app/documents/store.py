"""In-memory document and result store with TTL.

Phase 1 dev store, kept behind a small interface so it can move to Postgres + object
storage without touching callers. Nothing is written to disk.
"""

import asyncio
import time
import uuid

from loguru import logger

from app.config import get_settings
from app.documents import pdf
from app.documents.chunker import chunk_pages
from app.documents.language import detect_document_language, language_name
from app.documents.models import Document, DocumentStatus


class DocumentStore:
    def __init__(self, ttl_secs: int):
        self._ttl = ttl_secs
        self._docs: dict[str, Document] = {}
        self._results: dict[str, dict] = {}
        self._result_times: dict[str, float] = {}
        self._tasks: set[asyncio.Task] = set()

    # --- Documents ---

    async def ingest(self, *, session_id: str, filename: str, data: bytes) -> Document:
        """Validate synchronously (cheap checks), then parse in a worker thread."""
        settings = get_settings()
        doc = Document(
            doc_id=uuid.uuid4().hex,
            session_id=session_id,
            filename=filename,
            created_at=time.time(),
        )
        self._docs[doc.doc_id] = doc
        try:
            pdf.validate_bytes(data, max_mb=settings.max_pdf_mb)
        except pdf.PDFRejected as e:
            self._reject(doc, str(e))
            return doc
        task = asyncio.create_task(self._parse(doc, data))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return doc

    async def _parse(self, doc: Document, data: bytes) -> None:
        settings = get_settings()
        started = time.perf_counter()
        try:
            pages, warnings = await asyncio.to_thread(
                pdf.extract_pages,
                data,
                max_pages=settings.max_pages,
                min_chars_per_page=settings.min_chars_per_page,
            )
            if sum(len(p.text) for p in pages) > settings.max_doc_chars:
                raise pdf.PDFRejected("The document is too long to translate in one request.")

            dominant, per_page = await asyncio.to_thread(detect_document_language, [p.text for p in pages])
            for p in pages:
                p.language = per_page.get(p.number)
            other = sorted(n for n, lang in per_page.items() if lang != dominant)
            if other:
                warnings.append(f"Some pages appear to be in a different language: {other[:20]}.")
            if dominant and dominant not in settings.supported_languages:
                raise pdf.PDFRejected(
                    f"The document appears to be in {language_name(dominant)}. "
                    "Only English documents are supported right now."
                )

            doc.pages = pages
            doc.language = dominant
            doc.warnings = warnings
            doc.chunks = chunk_pages(pages, settings.chunk_target_chars)
            doc.status = DocumentStatus.READY
            logger.info(
                f"doc {doc.doc_id} ready: pages={doc.page_count} chars={doc.char_count} "
                f"chunks={len(doc.chunks)} lang={dominant} in {time.perf_counter() - started:.2f}s"
            )
        except pdf.PDFRejected as e:
            self._reject(doc, str(e))
        except Exception as e:  # never leave a document stuck in "processing"
            logger.exception(f"doc {doc.doc_id} parse failed")
            self._reject(doc, "The PDF could not be processed.")
            logger.debug(e)

    def _reject(self, doc: Document, reason: str) -> None:
        doc.status = DocumentStatus.REJECTED
        doc.error = reason
        logger.info(f"doc {doc.doc_id} rejected: {reason}")

    def get(self, doc_id: str) -> Document | None:
        self._expire()
        return self._docs.get(doc_id)

    def latest_for_session(self, session_id: str) -> Document | None:
        self._expire()
        docs = [d for d in self._docs.values() if d.session_id == session_id]
        return max(docs, key=lambda d: d.created_at) if docs else None

    # --- Results ---

    def put_result(self, result: dict) -> None:
        self._results[result["request_id"]] = result
        self._result_times[result["request_id"]] = time.time()

    def get_result(self, request_id: str) -> dict | None:
        self._expire()
        return self._results.get(request_id)

    def _expire(self) -> None:
        cutoff = time.time() - self._ttl
        for doc_id in [k for k, d in self._docs.items() if d.created_at < cutoff]:
            del self._docs[doc_id]
        for rid in [k for k, t in self._result_times.items() if t < cutoff]:
            self._results.pop(rid, None)
            self._result_times.pop(rid, None)


_store: DocumentStore | None = None


def get_store() -> DocumentStore:
    global _store
    if _store is None:
        _store = DocumentStore(get_settings().doc_ttl_secs)
    return _store
