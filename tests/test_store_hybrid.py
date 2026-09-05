from __future__ import annotations

import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

from docqa.config import Settings
from docqa.embeddings import StubEmbeddings
from docqa.store import HybridStore, ScoredChunk


def _reachable() -> bool:
    try:
        engine = create_engine(Settings().database_url, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


NEEDS_DB = pytest.mark.skipif(not _reachable(), reason="no reachable pgvector instance")

pytestmark = [pytest.mark.pg, NEEDS_DB]


def _store(collection: str) -> HybridStore:
    settings = Settings(collection_name=collection)
    return HybridStore(settings.database_url, settings.collection_name, StubEmbeddings(dim=4))


def test_store_roundtrip() -> None:
    store = _store("pytest_docqa_rt")
    store.reset()
    store.add_documents(
        [
            Document(
                page_content="Remote employees may work from anywhere in the country.",
                metadata={"source": "a.md", "chunk_index": 0},
            ),
            Document(
                page_content=(
                    "Working abroad over 30 days needs People team approval 45 days ahead."
                ),
                metadata={"source": "a.md", "chunk_index": 1},
            ),
        ]
    )
    assert store.count() == 2

    hits = store.hybrid_search("working abroad approval", top_k=2, vector_k=4, keyword_k=4)
    assert len(hits) == 2
    assert isinstance(hits[0], ScoredChunk)
    assert hits[0].rank == 1
    assert hits[1].rank == 2


def test_store_reset_clears() -> None:
    store = _store("pytest_docqa_reset")
    store.reset()
    store.add_documents(
        [Document(page_content="only one", metadata={"source": "b.md", "chunk_index": 0})]
    )
    assert store.count() == 1
    store.reset()
    assert store.count() == 0


def test_store_refuses_mismatched_dimension() -> None:
    settings = Settings(collection_name="pytest_docqa_dim")
    store = HybridStore(settings.database_url, settings.collection_name, StubEmbeddings(dim=4))
    store.reset()
    store.add_documents([Document(page_content="x", metadata={"source": "c.md", "chunk_index": 0})])

    other = HybridStore(settings.database_url, settings.collection_name, StubEmbeddings(dim=16))
    with pytest.raises(Exception):  # noqa: B017 - psycopg/pgvector raise type varies by driver
        other.hybrid_search("x", top_k=1)


def test_rrf_fuses_lists() -> None:
    a = Document(page_content="a", metadata={"chunk_id": "d1"})
    b = Document(page_content="b", metadata={"chunk_id": "d2"})
    scores = HybridStore._rrf([[a, b], [b]], k=60)
    assert scores["d2"] > scores["d1"]


def test_dedupe_by_chunk_id() -> None:
    a = Document(page_content="a", metadata={"chunk_id": "d1"})
    dup = Document(page_content="same", metadata={"chunk_id": "d1"})
    out = HybridStore._dedupe([a, dup])
    assert len(out) == 1
