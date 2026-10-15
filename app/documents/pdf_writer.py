"""Minimal dependency-free PDF writer, used for tests and sample documents only."""


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]], *, footer: str | None = None) -> bytes:
    """Build a PDF with one text line per list entry (Helvetica 11pt).

    An empty string leaves a blank line (paragraph break). ``footer`` is drawn on every
    page with the page number, to exercise header/footer stripping.
    """
    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_id = len(objects) + 1 + 2 * len(pages)  # reserved after page/content objects
    page_ids = []
    for n, lines in enumerate(pages, start=1):
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 720 Td"]
        for line in lines:
            ops.append(f"({_escape(line)}) Tj T*" if line else "T*")
        ops.append("ET")
        if footer:
            ops += ["BT", "/F1 9 Tf", "72 40 Td", f"({_escape(footer)}) Tj", "ET"]
            ops += ["BT", "/F1 9 Tf", "520 40 Td", f"({n}) Tj", "ET"]
        stream = "\n".join(ops).encode("latin-1")
        content_id = add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (pages_id, font_id, content_id)
            )
        )
    kids = " ".join(f"{i} 0 R" for i in page_ids).encode()
    assert add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))) == pages_id
    catalog_id = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog_id, xref)
    return bytes(out)
