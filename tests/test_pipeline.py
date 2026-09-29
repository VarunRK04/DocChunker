import pytest

from conftest import CORPUS, make_pdf
from docchunker.embeddings.hashing import HashingEmbedder
from docchunker.store.base import StoreMismatch
from docchunker.store.local import LocalStore


def test_ingest_search_reconstruct_delete(dc, corpus):
    results = dc.ingest_path(corpus)
    assert len(results) == 2 and not any(r.skipped for r in results)
    pets = next(r for r in results if r.name == "pets.txt")

    hits = dc.search("dog skin disease vet", k=3)
    assert hits[0].chunk.doc_id == pets.doc_id
    assert hits == sorted(hits, key=lambda h: -h.score)

    assert dc.reconstruct(pets.doc_id[:6]) == CORPUS["pets.txt"].strip()  # id prefixes work

    for r in results:
        assert dc.delete(r.doc_id) is True
    assert dc.documents() == []
    with pytest.raises(KeyError):
        dc.reconstruct(pets.doc_id)


def test_reingest_is_idempotent(dc, corpus):
    first = dc.ingest_file(corpus / "pets.txt")
    again = dc.ingest_file(corpus / "pets.txt")
    assert again.skipped and again.doc_id == first.doc_id and again.n_chunks == first.n_chunks
    assert len(dc.documents()) == 1
    forced = dc.ingest_file(corpus / "pets.txt", force=True)
    assert not forced.skipped and len(dc.documents()) == 1
    assert len(dc.store.get_chunks(first.doc_id)) == first.n_chunks  # no duplicate chunks


def test_search_filters(dc, corpus):
    dc.ingest_path(corpus)
    doc = next(d for d in dc.documents() if d.name == "cooking.txt")
    hits = dc.search("dog", k=5, doc_id=doc.doc_id)
    assert {h.chunk.doc_id for h in hits} == {doc.doc_id}
    assert dc.search("dog", kind="code") == []


def test_search_options_and_errors(dc, corpus):
    dc.ingest_path(corpus)
    assert dc.search("bread", k=2, hybrid=False)
    assert len(dc.search("bread flour dog", k=2, diverse=True)) == 2
    with pytest.raises(ValueError):
        dc.search("   ")
    with pytest.raises(KeyError):
        dc.search("x", doc_id="nope")


def test_pdf_pages_and_code(dc, tmp_path):
    pdf = make_pdf(tmp_path / "r.pdf", ["Intro page about dogs and vets. " * 8,
                                        "def add(a, b):\n    return a + b\n\nclass X:\n    pass"])
    r = dc.ingest_file(pdf)
    chunks = dc.store.get_chunks(r.doc_id)
    assert {c.page for c in chunks} == {1, 2}
    assert r.n_code_chunks >= 1
    assert dc.search("def add", kind="code")[0].chunk.page == 2


def test_blank_pdf_is_rejected(dc, tmp_path):
    pdf = make_pdf(tmp_path / "blank.pdf", [""])
    with pytest.raises(ValueError, match="no extractable text"):
        dc.ingest_file(pdf)
    assert dc.documents() == []


def test_folder_ingest_skips_bad_files(dc, corpus):
    (corpus / "empty.txt").write_text("   ")
    (corpus / "notes.xyz").write_text("ignored")
    errors = []
    results = dc.ingest_path(corpus, on_error=lambda f, e: errors.append(f.name))
    assert len(results) == 2 and errors == ["empty.txt"]


def test_unsupported_and_missing(dc, tmp_path):
    (tmp_path / "a.xyz").write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        dc.ingest_file(tmp_path / "a.xyz")
    with pytest.raises(FileNotFoundError):
        dc.ingest_path(tmp_path / "nothing-here")


def test_embedder_mismatch_detected(dc):
    dc.store.close()
    other = LocalStore(dc.settings.data_dir)
    with pytest.raises(StoreMismatch):
        other.bind("hashing-64", 64)
    other.bind(HashingEmbedder().name, HashingEmbedder().dim)  # same embedder is fine


def test_folder_ingest_skips_hidden_and_vendor_dirs(dc, corpus):
    for d in (".git", ".venv", "node_modules"):
        (corpus / d).mkdir()
        (corpus / d / "junk.md").write_text("should never be indexed")
    (corpus / "sub").mkdir()
    (corpus / "sub" / "deep.md").write_text("A nested markdown note about kittens.")
    names = sorted(r.name for r in dc.ingest_path(corpus))
    assert names == ["cooking.txt", "deep.md", "pets.txt"]


def test_embedder_is_lazy(tmp_path, corpus, monkeypatch):
    from docchunker import pipeline
    from docchunker.config import Settings

    calls = []
    monkeypatch.setattr(pipeline, "get_embedder", lambda s: calls.append(1) or HashingEmbedder())
    dc = pipeline.DocChunker(Settings(data_dir=tmp_path / "lazy", embedder="hashing"))
    assert dc.documents() == [] and calls == []  # listing needs no model
    dc.ingest_file(corpus / "pets.txt")
    dc.search("dog")
    assert calls == [1]  # loaded once, then reused


def test_docx_and_markdown(dc, tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Quarterly report on canine dermatology.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Metric", "Accuracy 91%"
    d.save(tmp_path / "r.docx")
    (tmp_path / "n.md").write_text("# Title\n\nSome *markdown* body about vaccines.\r\nSecond line.", encoding="utf-8")

    r1, r2 = dc.ingest_file(tmp_path / "r.docx"), dc.ingest_file(tmp_path / "n.md")
    assert "Accuracy 91%" in dc.reconstruct(r1.doc_id)
    assert "\r" not in dc.reconstruct(r2.doc_id)
