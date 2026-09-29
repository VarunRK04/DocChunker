import pytest

from docchunker.chunking import chunk_document, split_spans, stitch
from docchunker.models import Document, Page

LONG = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(200))


def _doc(text):
    return Document("t.txt", [Page(1, text)])


def test_spans_respect_size_and_cover_text():
    spans = split_spans(LONG, 200, 30)
    assert all(e - s <= 200 for s, e in spans)
    assert spans[0][0] == 0 and spans[-1][1] == len(LONG.rstrip())
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s2 <= e1  # contiguous or overlapping, never a gap
        assert s2 > s1


def test_overlap_exists_and_starts_on_word_boundary():
    spans = split_spans(LONG, 200, 40)
    overlaps = [e1 - s2 for (_, e1), (s2, _) in zip(spans, spans[1:])]
    assert any(o > 0 for o in overlaps)
    assert all(o <= 40 for o in overlaps)
    for s, _ in spans[1:]:
        assert LONG[s - 1].isspace()


def test_roundtrip_exact():
    chunks = chunk_document(_doc(LONG), "d", 200, 30)
    assert stitch(chunks) == LONG.strip()


def test_roundtrip_paragraphs_and_unicode():
    text = "Ünïcode — café ☕\n\n" + "para one. " * 60 + "\n\n" + "日本語のテキスト。" * 40
    chunks = chunk_document(_doc(text), "d", 120, 20)
    assert stitch(chunks) == text.strip()


def test_no_overlap_mode():
    chunks = chunk_document(_doc(LONG), "d", 200, 0)
    assert stitch(chunks) == LONG.strip()
    assert all(c.end <= n.start for c, n in zip(chunks, chunks[1:]))


def test_hard_split_without_separators():
    text = "x" * 1000
    assert stitch(chunk_document(_doc(text), "d", 100, 10)) == text


def test_empty_and_blank():
    assert chunk_document(_doc(""), "d") == []
    assert chunk_document(_doc("  \n\n \t"), "d") == []


def test_page_numbers():
    doc = Document("t.pdf", [Page(1, "alpha " * 50), Page(2, "beta " * 50)])
    chunks = chunk_document(doc, "d", 100, 10)
    assert chunks[0].page == 1 and chunks[-1].page == 2
    assert [c.page for c in chunks] == sorted(c.page for c in chunks)


def test_ids_stable_and_ordered():
    a = chunk_document(_doc(LONG), "abc", 200, 30)
    b = chunk_document(_doc(LONG), "abc", 200, 30)
    assert [c.id for c in a] == [c.id for c in b]
    assert a[0].id == "abc:000000"


@pytest.mark.parametrize("size,overlap", [(10, 0), (100, 60), (100, -1)])
def test_bad_params(size, overlap):
    with pytest.raises(ValueError):
        split_spans("hello world " * 20, size, overlap)
