from __future__ import annotations

from docqa.config import Settings


def test_defaults() -> None:
    s = Settings(embedding_provider="stub", llm_provider="stub", reranker="none")
    assert s.embedding_dim == 384
    assert s.top_k == 4
    assert s.vector_k == 8
    assert s.keyword_k == 8
    assert s.llm_temperature == 0.0


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("DOCQA_TOP_K", "6")
    monkeypatch.setenv("DOCQA_LLM_PROVIDER", "stub")
    monkeypatch.setenv("DOCQA_EMBEDDING_PROVIDER", "stub")
    s = Settings()
    assert s.top_k == 6
    assert s.llm_provider == "stub"


def test_env_prefix_is_docqa(monkeypatch) -> None:
    monkeypatch.setenv("DOCQA_COLLECTION_NAME", "custom-policies")
    monkeypatch.setenv("DOCQA_EMBEDDING_PROVIDER", "stub")
    monkeypatch.setenv("DOCQA_LLM_PROVIDER", "stub")
    assert Settings().collection_name == "custom-policies"
