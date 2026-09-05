from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.documents import Document

from docqa.config import Settings
from docqa.graph import RagAgent
from docqa.llm import StubChatModel
from docqa.reranker import PassThroughReranker
from docqa.store import ScoredChunk


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "policies"


def make_settings(**overrides) -> Settings:
    defaults = dict(
        embedding_provider="stub",
        llm_provider="stub",
        reranker="none",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


class FakeStore:
    """In-memory stand-in for HybridStore: enough surface for graph/API tests."""

    def __init__(self, docs: Sequence[Document] | None = None) -> None:
        self.docs = list(docs or [])
        self.engine = None

    def hybrid_search(self, question, vector_k=8, keyword_k=8, top_k=4):
        out = []
        for i, doc in enumerate(self.docs[:top_k]):
            out.append(ScoredChunk(document=doc, score=1.0 / (i + 1), rank=i + 1))
        return out


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="Remote employees may work from anywhere in the country.",
            metadata={"source": "remote-work.md", "chunk_index": 0, "chunk_id": "a"},
        ),
        Document(
            page_content="Working abroad over 30 days needs People team approval 45 days ahead.",
            metadata={"source": "remote-work.md", "chunk_index": 1, "chunk_id": "b"},
        ),
    ]


@pytest.fixture
def agent(settings: Settings, sample_docs: list[Document]) -> RagAgent:
    return RagAgent(
        store=FakeStore(sample_docs),
        llm=StubChatModel(),
        reranker=PassThroughReranker(),
        settings=settings,
    )
