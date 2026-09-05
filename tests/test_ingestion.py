from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from docqa.chunking import ingest
from docqa.chunking_eval import load_eval_set


class RecordingStore:
    def __init__(self) -> None:
        self.added: list[Document] = []
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True

    def add_documents(self, docs) -> list[str]:
        self.added.extend(docs)
        return [f"id-{i}" for i in range(len(docs))]


def test_ingest_chunks_and_records(data_dir: Path) -> None:
    store = RecordingStore()
    count = ingest(data_dir, chunk_size=256, chunk_overlap=32, store=store, reset=True)

    assert count == len(store.added)
    assert store.reset_called is True
    assert all("source" in d.metadata and "chunk_index" in d.metadata for d in store.added)


def test_ingest_without_reset() -> None:
    store = RecordingStore()
    ingest(Path("data/policies"), chunk_size=512, chunk_overlap=64, store=store, reset=False)
    assert store.reset_called is False
    assert store.added


@pytest.mark.parametrize("source", ["remote-work-policy.md", "ai-usage-policy.md"])
def test_ingest_labels_source(data_dir: Path, source: str) -> None:
    store = RecordingStore()
    ingest(data_dir, chunk_size=512, chunk_overlap=64, store=store)
    assert any(d.metadata.get("source") == source for d in store.added)


def test_eval_set_loads(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text('{"question": "q1", "gold_source": "a.md"}\n', encoding="utf-8")
    items = load_eval_set(path)
    assert items == [{"question": "q1", "gold_source": "a.md"}]
