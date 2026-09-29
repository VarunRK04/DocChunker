from __future__ import annotations

from pathlib import Path

from ..models import Document
from .docx import load_docx
from .pdf import load_pdf
from .text import load_text

LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
    ".markdown": load_text,
    ".rst": load_text,
    ".docx": load_docx,
}

SUPPORTED = tuple(LOADERS)


def load_document(path: str | Path) -> Document:
    p = Path(path)
    loader = LOADERS.get(p.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type {p.suffix!r} (supported: {', '.join(SUPPORTED)})")
    if not p.is_file():
        raise FileNotFoundError(p)
    return loader(p)
