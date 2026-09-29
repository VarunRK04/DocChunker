from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class Document:
    name: str
    pages: list[Page]
    path: str = ""

    @property
    def text(self) -> str:
        return PAGE_SEP.join(p.text for p in self.pages)


PAGE_SEP = "\n\n"


@dataclass
class Chunk:
    id: str
    doc_id: str
    index: int
    text: str
    start: int  # char offsets into Document.text
    end: int
    page: int
    kind: str = "text"  # "text" | "code"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    vector: "object | None" = field(default=None, repr=False)


@dataclass
class DocInfo:
    doc_id: str
    name: str
    path: str
    n_chunks: int
    n_chars: int
    created: float
