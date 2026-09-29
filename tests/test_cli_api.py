import json

import pytest

from docchunker.cli import main


def _run(capsys, tmp_path, *args):
    code = main(["--data-dir", str(tmp_path / "d"), "--embedder", "hashing", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_flow(capsys, corpus, tmp_path):
    code, out, _ = _run(capsys, tmp_path, "ingest", str(corpus))
    assert code == 0 and "pets.txt" in out
    code, out, _ = _run(capsys, tmp_path, "ingest", str(corpus))
    assert "already indexed" in out

    code, out, _ = _run(capsys, tmp_path, "search", "sourdough starter", "--json", "-k", "2")
    data = json.loads(out)
    assert code == 0 and data[0]["text"].startswith("To bake")

    code, out, _ = _run(capsys, tmp_path, "docs", "--json")
    doc = next(d for d in json.loads(out) if d["name"] == "pets.txt")
    code, out, _ = _run(capsys, tmp_path, "reconstruct", doc["doc_id"][:8])
    assert "vaccinations" in out
    code, out, _ = _run(capsys, tmp_path, "delete", doc["doc_id"])
    assert "deleted" in out


def test_cli_errors(capsys, tmp_path):
    code, _, err = _run(capsys, tmp_path, "reconstruct", "zzz")
    assert code == 1 and "No document" in err
    code, _, err = _run(capsys, tmp_path, "ingest", str(tmp_path / "missing.pdf"))
    assert code == 1 and "error" in err


def test_api(dc, corpus):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from docchunker.api import create_app

    c = TestClient(create_app(dc=dc))
    assert c.get("/health").json() == {"status": "ok"}
    body = (corpus / "pets.txt").read_bytes()
    r = c.post("/ingest", files={"file": ("pets.txt", body)})
    assert r.status_code == 200 and not r.json()["skipped"]
    assert c.post("/ingest", files={"file": ("pets.txt", body)}).json()["skipped"]
    assert c.post("/ingest", files={"file": ("x.exe", b"MZ")}).status_code == 415
    assert c.post("/ingest", files={"file": ("e.txt", b"  ")}).status_code == 422

    hits = c.get("/search", params={"q": "dog vaccinations", "k": 2}).json()
    assert hits and "vaccinations" in hits[0]["text"]
    assert c.get("/search", params={"q": "x", "kind": "bogus"}).status_code == 422
    doc_id = c.get("/docs-index").json()[0]["doc_id"]
    assert "Dogs are loyal" in c.get(f"/docs-index/{doc_id}/text").json()["text"]
    assert c.delete(f"/docs-index/{doc_id}").status_code == 200
    assert c.get(f"/docs-index/{doc_id}/text").status_code == 404


def test_api_rejects_oversized_upload(dc, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from docchunker import api

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    c = TestClient(api.create_app(dc=dc))
    assert c.post("/ingest", files={"file": ("big.txt", b"x" * 11)}).status_code == 413
