from __future__ import annotations

from pathlib import Path

from ..models import Document, Page


def _read(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_text(path: Path) -> Document:
    text = _read(path).replace("\r\n", "\n").replace("\r", "\n")  # same text on every OS
    return Document(name=path.name, pages=[Page(1, text)], path=str(path))
