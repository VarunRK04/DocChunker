from __future__ import annotations

from pathlib import Path

from ..models import Document, Page


def load_pdf(path: Path) -> Document:
    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        pages = [Page(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    return Document(name=path.name, pages=pages, path=str(path))


def extract_images(path: str | Path, out_dir: str | Path) -> list[Path]:
    """Save every embedded image of a PDF to out_dir; return the written paths."""
    import fitz

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[int] = set()
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            for n, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                if xref in seen:  # same image reused on many pages
                    continue
                seen.add(xref)
                info = doc.extract_image(xref)
                target = out / f"image_p{page_no}_{n}.{info['ext']}"
                target.write_bytes(info["image"])
                written.append(target)
    return written
