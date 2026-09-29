from __future__ import annotations

from pathlib import Path

from ..models import Document, Page


def load_docx(path: Path) -> Document:
    try:
        import docx
    except ImportError as e:
        raise RuntimeError("Reading .docx needs python-docx: pip install docchunker[docx]") from e
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return Document(name=path.name, pages=[Page(1, "\n".join(parts))], path=str(path))
