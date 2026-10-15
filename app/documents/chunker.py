"""Split a document into translation chunks on paragraph boundaries."""

import re

from app.documents.models import Chunk, Page

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_pages(pages: list[Page], target_chars: int) -> list[Chunk]:
    """Greedy packing of paragraphs into chunks of about ``target_chars``.

    Paragraphs are never split unless a single paragraph exceeds the target, in which
    case it is split on sentence boundaries. Each chunk records the pages it covers.
    """
    chunks: list[Chunk] = []
    parts: list[str] = []
    pages_in: list[int] = []
    size = 0

    def flush():
        nonlocal parts, pages_in, size
        if parts:
            chunks.append(Chunk(index=len(chunks), pages=sorted(set(pages_in)), text="\n\n".join(parts)))
        parts, pages_in, size = [], [], 0

    for page in pages:
        for block in page.blocks:
            for piece in _split_long(block.text, target_chars):
                if size and size + len(piece) > target_chars:
                    flush()
                parts.append(piece)
                pages_in.append(page.number)
                size += len(piece) + 2
    flush()
    return chunks


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for sentence in _SENTENCE_RE.split(text):
        if cur and len(cur) + len(sentence) + 1 > limit:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}" if cur else sentence
        while len(cur) > limit:  # a single "sentence" longer than the limit
            out.append(cur[:limit])
            cur = cur[limit:]
    if cur:
        out.append(cur)
    return out
