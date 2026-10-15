import pytest

from app.documents import pdf
from app.documents.chunker import chunk_pages
from app.documents.language import detect_document_language
from app.documents.models import BlockType
from app.documents.pdf_writer import make_pdf

PARA = [
    "The quarterly budget for 2026 is 4.2 million dollars, an increase of",
    "12 percent over the previous quarter. Most of the new spending goes to",
    "hiring engineers and expanding the data center in Pune.",
]


def _pages(n: int) -> list[list[str]]:
    return [["Section " + str(i), ""] + PARA + [""] + PARA for i in range(1, n + 1)]


def test_validate_rejects_non_pdf_and_empty():
    with pytest.raises(pdf.PDFRejected, match="not a PDF"):
        pdf.validate_bytes(b"hello world", max_mb=1)
    with pytest.raises(pdf.PDFRejected, match="empty"):
        pdf.validate_bytes(b"", max_mb=1)


def test_validate_rejects_oversize():
    with pytest.raises(pdf.PDFRejected, match="larger than"):
        pdf.validate_bytes(b"%PDF-" + b"0" * (1024 * 1024), max_mb=1)


def test_extract_text_headings_paragraphs_and_strips_footer():
    data = make_pdf(_pages(4), footer="ACME Confidential")
    pages, warnings = pdf.extract_pages(data, max_pages=10, min_chars_per_page=25)
    assert len(pages) == 4 and warnings == []
    first = pages[0]
    assert first.blocks[0].type == BlockType.HEADING and first.blocks[0].text == "Section 1"
    assert first.blocks[1].type == BlockType.PARAGRAPH
    assert first.blocks[1].text.startswith("The quarterly budget for 2026 is 4.2 million dollars, an increase of 12")
    joined = "\n".join(p.text for p in pages)
    assert "ACME Confidential" not in joined  # repeated footer removed
    assert "\n4\n" not in joined


def test_extract_rejects_scanned_like_pdf():
    data = make_pdf([[""], [""], [""]])
    with pytest.raises(pdf.PDFRejected, match="scanned"):
        pdf.extract_pages(data, max_pages=10, min_chars_per_page=25)


def test_extract_rejects_too_many_pages():
    with pytest.raises(pdf.PDFRejected, match="limit is 2"):
        pdf.extract_pages(make_pdf(_pages(3)), max_pages=2, min_chars_per_page=25)


def test_extract_rejects_garbage_after_magic():
    with pytest.raises(pdf.PDFRejected, match="damaged"):
        pdf.extract_pages(b"%PDF-1.4 not really a pdf", max_pages=10, min_chars_per_page=25)


def test_chunker_packs_paragraphs_and_tracks_pages():
    pages, _ = pdf.extract_pages(make_pdf(_pages(6)), max_pages=10, min_chars_per_page=25)
    chunks = chunk_pages(pages, target_chars=700)
    assert len(chunks) > 1
    assert all(len(c.text) <= 700 for c in chunks)
    assert chunks[0].pages[0] == 1 and chunks[-1].pages[-1] == 6
    # No content lost or duplicated
    assert sum(len(c.text.replace("\n\n", "")) for c in chunks) == sum(
        len(b.text) for p in pages for b in p.blocks
    )


def test_chunker_splits_oversized_paragraph_on_sentences():
    from app.documents.models import Block, Page

    long = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_pages([Page(1, [Block(BlockType.PARAGRAPH, long)])], target_chars=500)
    assert len(chunks) > 1 and all(len(c.text) <= 500 for c in chunks)


def test_document_language_detection():
    pages, _ = pdf.extract_pages(make_pdf(_pages(2)), max_pages=10, min_chars_per_page=25)
    dominant, per_page = detect_document_language([p.text for p in pages])
    assert dominant == "en" and per_page == {1: "en", 2: "en"}


def test_sample_like_document_keeps_paragraphs_and_strips_two_page_footer():
    from scripts.make_sample_pdf import PAGES

    pages, _ = pdf.extract_pages(make_pdf(PAGES, footer="Internal - Planning"), max_pages=10, min_chars_per_page=25)
    first = [(b.type, b.text[:20]) for b in pages[0].blocks]
    assert first[0] == (BlockType.HEADING, "Q3 2026 Operating Pl")
    assert (BlockType.HEADING, "Hiring") in first
    assert len(pages[0].blocks) == 4
    assert all("Internal" not in b.text for p in pages for b in p.blocks)
