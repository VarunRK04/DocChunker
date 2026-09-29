import pytest

from docchunker.config import Settings
from docchunker.embeddings.hashing import HashingEmbedder
from docchunker.pipeline import DocChunker
from docchunker.store.local import LocalStore

CORPUS = {
    "pets.txt": (
        "Dogs are loyal companions. A healthy dog needs daily exercise, fresh water and regular "
        "vaccinations against rabies and distemper.\n\nSkin infections in dogs often show up as itching, "
        "redness and hair loss. A veterinarian can diagnose the disease from a photo or a physical exam."
    ),
    "cooking.txt": (
        "To bake sourdough bread you need flour, water, salt and an active starter.\n\n"
        "Let the dough ferment overnight, shape it, and bake in a hot dutch oven until the crust is golden."
    ),
}


def make_pdf(path, pages):
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def dc(tmp_path):
    settings = Settings(store="local", data_dir=tmp_path / "data", embedder="hashing",
                        chunk_size=200, chunk_overlap=30)
    return DocChunker(settings, store=LocalStore(settings.data_dir), embedder=HashingEmbedder())


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    for name, body in CORPUS.items():
        (folder / name).write_text(body, encoding="utf-8")
    return folder
