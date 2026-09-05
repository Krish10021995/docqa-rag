from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from docqa.chunking import load_documents, split_documents


def test_load_documents(data_dir: Path) -> None:
    docs = load_documents(data_dir)
    assert len(docs) >= 3
    assert {d.metadata["source"] for d in docs} >= {
        "remote-work-policy.md",
        "ai-usage-policy.md",
        "onboarding-guide.md",
    }
    assert all(d.page_content for d in docs)


def test_split_documents_metadata() -> None:
    pages = "\n\n".join("# Section a word longer" for _ in range(30))
    doc = Document(page_content=pages, metadata={"source": "x.md"})
    chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.metadata["source"] == "x.md"
        assert c.metadata["chunk_index"] == i
        assert len(c.page_content) <= 220


def test_split_documents_smoke() -> None:
    docs = load_documents(Path("data/policies"))
    chunks = split_documents(docs, chunk_size=402, chunk_overlap=80)
    assert chunks
    assert all("source" in c.metadata and "chunk_index" in c.metadata for c in chunks)
