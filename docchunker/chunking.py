"""Recursive character splitter that keeps exact offsets into the source text.

Because every chunk records ``start``/``end`` in the document text, the original
can be rebuilt exactly (see :func:`stitch`) even though neighbouring chunks overlap.
"""
from __future__ import annotations

import bisect
import hashlib

from .classify import classify
from .models import PAGE_SEP, Chunk, Document

_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def _atoms(text: str, start: int, end: int, seps: tuple[str, ...], size: int) -> list[tuple[int, int]]:
    """Split text[start:end] into contiguous spans of at most `size` chars."""
    if end - start <= size:
        return [(start, end)]
    if not seps:
        return [(i, min(i + size, end)) for i in range(start, end, size)]
    sep, rest = seps[0], seps[1:]
    spans: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        j = text.find(sep, pos, end)
        cut = end if j == -1 else j + len(sep)  # keep the separator with the left piece
        spans.extend(_atoms(text, pos, cut, rest, size) if cut - pos > size else [(pos, cut)])
        pos = cut
    return spans


def _snap_forward(text: str, pos: int, limit: int) -> int:
    """Move pos forward to just after the next whitespace so overlaps start on a word."""
    if pos <= 0:
        return 0
    if text[pos - 1].isspace():
        return pos
    for i in range(pos, limit):
        if text[i].isspace():
            return i + 1
    return pos


def _split_range(text: str, lo: int, hi: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Chunk text[lo:hi] into contiguous groups, each extended backwards by a word-aligned overlap.

    Interior chunks keep their raw whitespace so the source can be rebuilt exactly.
    """
    core = chunk_size - overlap  # room left once the overlap prefix is added
    groups: list[tuple[int, int]] = []
    cur_s = cur_e = None
    for s, e in _atoms(text, lo, hi, _SEPARATORS, core):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif e - cur_s <= core:
            cur_e = e
        else:
            groups.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        groups.append((cur_s, cur_e))

    spans = []
    for i, (s, e) in enumerate(groups):
        if i and overlap:
            s = _snap_forward(text, max(groups[i - 1][1] - overlap, groups[i - 1][0]), s)
        spans.append((s, e))
    return spans


def _check(chunk_size: int, overlap: int) -> None:
    if chunk_size < 20:
        raise ValueError("chunk_size must be >= 20")
    if not 0 <= overlap <= chunk_size // 2:
        raise ValueError("overlap must be between 0 and chunk_size/2")


def split_spans(text: str, chunk_size: int = 800, overlap: int = 100) -> list[tuple[int, int]]:
    """Chunk a whole text; the result covers text.strip() exactly."""
    _check(chunk_size, overlap)
    lo = len(text) - len(text.lstrip())
    hi = len(text.rstrip())
    return _split_range(text, lo, hi, chunk_size, overlap) if hi > lo else []


def make_doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def chunk_document(doc: Document, doc_id: str, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Chunk page by page so every chunk belongs to exactly one page (accurate citations)."""
    _check(chunk_size, overlap)
    text = doc.text
    spans: list[tuple[int, int, int]] = []  # start, end, page
    off = 0
    for page in doc.pages:
        end = off + len(page.text)
        if page.text.strip():
            spans.extend((s, e, page.number) for s, e in _split_range(text, off, end, chunk_size, overlap))
        off = end + len(PAGE_SEP)
    if not spans:
        return []
    # the document as a whole is compared stripped: trim its first and last chunk only
    s0, e0, p0 = spans[0]
    spans[0] = (s0 + len(text[s0:e0]) - len(text[s0:e0].lstrip()), e0, p0)
    s1, e1, p1 = spans[-1]
    spans[-1] = (s1, s1 + len(text[s1:e1].rstrip()), p1)

    return [
        Chunk(id=f"{doc_id}:{i:06d}", doc_id=doc_id, index=i, text=text[s:e], start=s, end=e, page=page,
              kind=classify(text[s:e]))
        for i, (s, e, page) in enumerate(spans)
    ]


def stitch(chunks: list[Chunk]) -> str:
    """Rebuild the source text from (possibly overlapping) chunks."""
    out: list[str] = []
    covered = None
    for c in sorted(chunks, key=lambda c: c.index):
        if covered is None:
            out.append(c.text)
        elif c.start >= covered:
            out.append(text_gap(c.start - covered) + c.text)
        else:
            out.append(c.text[covered - c.start :])
        covered = max(c.end, covered or 0)
    return "".join(out)


def text_gap(n: int) -> str:
    # dropped whitespace between chunks; original bytes are unknown, use a blank line or space
    return "" if n <= 0 else "\n\n" if n >= 2 else " "
