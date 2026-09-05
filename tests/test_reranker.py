from __future__ import annotations

from langchain_core.documents import Document

from docqa.config import Settings
from docqa.reranker import CrossEncoderReranker, PassThroughReranker, create_reranker


def docs(n: int) -> list[Document]:
    return [Document(page_content=f"chunk {i}", metadata={"chunk_id": str(i)}) for i in range(n)]


def test_pass_through_keeps_order():
    r = PassThroughReranker()
    result = r.rerank("q", docs(4), top_k=4)
    assert [d.page_content for d, _ in result] == [f"chunk {i}" for i in range(4)]


def test_pass_through_respects_top_k():
    r = PassThroughReranker()
    assert len(r.rerank("q", docs(10), top_k=3)) == 3


def test_pass_through_empty():
    assert PassThroughReranker().rerank("q", [], top_k=4) == []


def test_create_reranker_default_is_pass_through():
    assert isinstance(create_reranker(Settings(reranker="none")), PassThroughReranker)


def test_cross_encoder_reranker_requires_extra():
    try:
        CrossEncoderReranker("unused/model")
    except RuntimeError as exc:
        assert "local" in str(exc)
    else:  # pragma: no cover - only when sentence-transformers is installed
        pass
