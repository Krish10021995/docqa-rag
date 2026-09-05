from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from docqa.api import create_app
from docqa.graph import RagAgent
from docqa.llm import StubChatModel
from docqa.reranker import PassThroughReranker
from tests.conftest import FakeStore, make_settings


@pytest.fixture
def client() -> TestClient:
    settings = make_settings()
    agent = RagAgent(
        store=FakeStore(),
        llm=StubChatModel(),
        reranker=PassThroughReranker(),
        settings=settings,
    )
    return TestClient(create_app(agent, settings))


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_readyz_reports_down_when_no_database(client: TestClient) -> None:
    res = client.get("/readyz")
    assert res.status_code == 200
    assert res.json()["database"] == "down"


def test_ask_returns_no_context_when_empty(client: TestClient) -> None:
    res = client.post("/ask", json={"question": "How long can I work abroad?"})
    assert res.status_code == 200
    body = res.json()
    assert "could not find" in body["answer"].lower()
    assert body["sources"] == []


def test_ask_returns_citations_with_context(client: TestClient, sample_docs) -> None:
    settings = make_settings()
    agent = RagAgent(
        store=FakeStore(sample_docs),
        llm=StubChatModel(),
        reranker=PassThroughReranker(),
        settings=settings,
    )
    c = TestClient(create_app(agent, settings))
    res = c.post("/ask", json={"question": "Work abroad?"})
    assert res.status_code == 200
    body = res.json()
    assert "stub answer" in body["answer"]
    assert len(body["sources"]) == 2
    assert {s["rank"] for s in body["sources"]} == {1, 2}


def test_ask_rejects_empty_question(client: TestClient) -> None:
    res = client.post("/ask", json={"question": ""})
    assert res.status_code == 422


def test_ask_stream_emits_events(client: TestClient) -> None:
    res = client.post("/ask/stream", json={"question": "Work abroad?"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = [
        json.loads(line[6:])
        for block in res.text.split("\n\n")
        for line in block.splitlines()
        if line.startswith("data: ")
    ]
    assert events, "must emit at least one SSE event"

    kinds = {tuple(e.keys()) for e in events}
    assert ("node", "chunks") in kinds
    assert ("answer",) in kinds
    assert ("ok",) in kinds


def test_stream_reports_retrieve_count(client: TestClient, monkeypatch) -> None:
    from tests.conftest import FakeStore

    store = FakeStore()

    def fake_search(question, vector_k=8, keyword_k=8, top_k=4):
        return []

    store.hybrid_search = fake_search
    agent = RagAgent(
        store=store,
        llm=StubChatModel(),
        reranker=PassThroughReranker(),
        settings=make_settings(),
    )
    c = TestClient(create_app(agent, make_settings()))
    res = c.post("/ask/stream", json={"question": "anything"})
    events = [
        json.loads(line[6:])
        for block in res.text.split("\n\n")
        for line in block.splitlines()
        if line.startswith("data: ")
    ]
    status = next(e for e in events if "chunks" in e)
    assert status["chunks"] == 0
