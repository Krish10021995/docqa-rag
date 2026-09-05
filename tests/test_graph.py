from __future__ import annotations

from docqa.graph import RagAgent
from docqa.llm import StubChatModel
from docqa.reranker import PassThroughReranker
from tests.conftest import FakeStore, make_settings


def test_ask_returns_answer_with_citations(agent: RagAgent) -> None:
    result = agent.ask("How long can I work from abroad?")
    assert "stub answer" in result["answer"]
    assert result["sources"], "citations [0,1] should map to the two fake contexts"
    ranks = {s["rank"] for s in result["sources"]}
    assert ranks == {1, 2}
    assert result["sources"][0]["source"] == "remote-work.md"


def test_no_context_answer_when_store_empty() -> None:
    agent = RagAgent(
        store=FakeStore([]),
        llm=StubChatModel(),
        reranker=PassThroughReranker(),
        settings=make_settings(),
    )
    result = agent.ask("anything")
    assert "could not find" in result["answer"].lower()
    assert result["sources"] == []


def test_stream_emits_retrieve_and_answer(agent: RagAgent) -> None:
    updates = list(agent.stream("How about working abroad?"))
    seen = {next(iter(u)) for u in updates}
    assert "retrieve" in seen
    assert "generate" in seen
