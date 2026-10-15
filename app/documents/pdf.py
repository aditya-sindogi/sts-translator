"""PDF validation and text extraction.

Uses pypdfium2 (Apache-2.0/BSD). PyMuPDF is deliberately avoided because it is AGPL.
Scanned PDFs (no text layer) are rejected in Phase 1; OCR comes in Phase 2.
"""

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

import pypdfium2 as pdfium

from app.documents.models import Block, BlockType, Page

PDF_MAGIC = b"%PDF-"
_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)
_WS_RE = re.compile("[ \t\u00a0]+")


class PDFRejected(Exception):
    """The PDF cannot be processed. The message is safe to show and speak to the user."""


def validate_bytes(data: bytes, *, max_mb: int) -> None:
    if len(data) == 0:
        raise PDFRejected("The file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise PDFRejected(f"The file is larger than {max_mb} MB.")
    if not data[:1024].lstrip().startswith(PDF_MAGIC):
        raise PDFRejected("The file is not a PDF.")


def extract_pages(data: bytes, *, max_pages: int, min_chars_per_page: int) -> tuple[list[Page], list[str]]:
    """Extract normalized pages. Returns (pages, warnings). Raises PDFRejected."""
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        if "password" in str(e).lower():
            raise PDFRejected("The PDF is password-protected.") from e
        raise PDFRejected("The PDF is damaged or unreadable.") from e

    try:
        n = len(pdf)
        if n == 0:
            raise PDFRejected("The PDF has no pages.")
        if n > max_pages:
            raise PDFRejected(f"The PDF has {n} pages; the limit is {max_pages}.")

        raw_pages: list[list[_Line]] = []
        heights: list[float] = []
        for i in range(n):
            page = pdf[i]
            heights.append(page.get_height())
            textpage = page.get_textpage()
            raw_pages.append(_page_lines(textpage))
            textpage.close()
            page.close()
    finally:
        pdf.close()

    total_chars = sum(len(line.text) for lines in raw_pages for line in lines)
    if total_chars < min_chars_per_page * max(1, n // 2):
        raise PDFRejected(
            "The PDF has no readable text layer, it looks scanned. Scanned PDFs are not supported yet."
        )

    raw_pages = _strip_repeated_margins(raw_pages, heights)

    warnings: list[str] = []
    sparse = [i + 1 for i, lines in enumerate(raw_pages) if sum(len(x.text) for x in lines) < min_chars_per_page]
    if sparse:
        warnings.append(f"Pages with little or no text (possibly images): {_ranges(sparse)}.")

    pages = [Page(number=i + 1, blocks=_to_blocks(_with_paragraph_breaks(lines))) for i, lines in enumerate(raw_pages)]
    return pages, warnings


@dataclass
class _Line:
    text: str
    left: float
    bottom: float
    top: float


def _page_lines(textpage) -> list[_Line]:
    """Rebuild visual lines from pdfium text rects (merging rects on the same baseline)."""
    pieces: list[_Line] = []
    for i in range(textpage.count_rects()):
        left, bottom, right, top = textpage.get_rect(i)
        text = textpage.get_text_bounded(left, bottom, right, top).replace("\x00", "")
        text = _WS_RE.sub(" ", text.replace("\r", " ").replace("\n", " ")).strip()
        if text:
            pieces.append(_Line(text, left, bottom, top))
    # Top-to-bottom, then left-to-right; merge pieces whose vertical extents overlap.
    pieces.sort(key=lambda p: (-p.top, p.left))
    lines: list[_Line] = []
    for piece in pieces:
        last = lines[-1] if lines else None
        if last and min(last.top, piece.top) - max(last.bottom, piece.bottom) > 0.5 * (piece.top - piece.bottom):
            last.text = f"{last.text} {piece.text}" if piece.left >= last.left else f"{piece.text} {last.text}"
            last.left = min(last.left, piece.left)
            last.bottom, last.top = min(last.bottom, piece.bottom), max(last.top, piece.top)
        else:
            lines.append(_Line(piece.text, piece.left, piece.bottom, piece.top))
    return lines


def _with_paragraph_breaks(lines: list[_Line]) -> list[str]:
    """Insert blank lines where the vertical gap is clearly larger than normal leading."""
    if not lines:
        return []
    gaps = [prev.bottom - cur.top for prev, cur in zip(lines, lines[1:])]
    line_h = median(x.top - x.bottom for x in lines)
    # Normal leading is the low end of the gap distribution; paragraph gaps sit above it.
    # (A median breaks on documents with many short paragraphs.)
    positive = sorted(g for g in gaps if g >= 0)
    normal_gap = positive[len(positive) // 4] if positive else 0.0
    threshold = max(normal_gap * 1.8 + 1.0, 0.6 * line_h)
    out = [lines[0].text]
    for gap, line in zip(gaps, lines[1:]):
        if gap > threshold:
            out.append("")
        out.append(line.text)
    return out


def _strip_repeated_margins(pages: list[list[_Line]], heights: list[float]) -> list[list[_Line]]:
    """Drop page numbers and header/footer lines that repeat (digits ignored) in the margins."""

    def in_margin(line: _Line, h: float) -> bool:
        return line.bottom < 0.07 * h or line.top > 0.94 * h

    def key(text: str, page_no: int) -> str:
        # Only the page's own number is normalized ("Page 3 of 10" -> "page # of 10"),
        # so headings like "Section 2" are not mistaken for running headers.
        return re.sub(rf"(?<!\d){page_no}(?!\d)", "#", text.lower())

    counts: Counter[str] = Counter()
    for n, (lines, h) in enumerate(zip(pages, heights), start=1):
        counts.update({key(x.text, n) for x in lines if in_margin(x, h)})
    threshold = max(2, int(len(pages) * 0.6))
    repeated = {k for k, c in counts.items() if c >= threshold} if len(pages) >= 2 else set()

    return [
        [
            x
            for x in lines
            if not (in_margin(x, h) and (key(x.text, n) in repeated or _PAGE_NUMBER_RE.match(x.text)))
        ]
        for n, (lines, h) in enumerate(zip(pages, heights), start=1)
    ]


def _to_blocks(lines: list[str]) -> list[Block]:
    """Group lines into paragraphs (blank-line separated), re-flowing hard line wraps."""
    blocks: list[Block] = []
    current: list[str] = []

    def flush():
        if current:
            text = _reflow(current)
            btype = BlockType.HEADING if _looks_like_heading(current) else BlockType.PARAGRAPH
            blocks.append(Block(type=btype, text=text))
            current.clear()

    for line in lines:
        if not line:
            flush()
        else:
            current.append(line)
    flush()
    return blocks


def _reflow(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if not text:
            text = line
        elif text.endswith("-") and line[:1].islower():
            text = text[:-1] + line  # de-hyphenate a wrapped word
        else:
            text += " " + line
    return text


def _looks_like_heading(lines: list[str]) -> bool:
    if len(lines) != 1:
        return False
    line = lines[0]
    return len(line) <= 80 and not line.endswith((".", ",", ";", ":")) and len(line.split()) <= 12


def _ranges(nums: list[int]) -> str:
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(parts)
