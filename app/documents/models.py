"""Normalized document structures shared by parsing, chunking and translation."""

from dataclasses import dataclass, field
from enum import StrEnum


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"


@dataclass
class Block:
    type: BlockType
    text: str


@dataclass
class Page:
    number: int  # 1-based
    blocks: list[Block]
    language: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass
class Chunk:
    index: int
    pages: list[int]
    text: str


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    REJECTED = "rejected"


@dataclass
class Document:
    doc_id: str
    session_id: str
    filename: str
    status: DocumentStatus = DocumentStatus.PROCESSING
    pages: list[Page] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    language: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: float = 0.0

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    def summary(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "status": self.status.value,
            "pages": self.page_count,
            "chars": self.char_count,
            "chunks": len(self.chunks),
            "language": self.language,
            "warnings": self.warnings,
            "error": self.error,
        }
