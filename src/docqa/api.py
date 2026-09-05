from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text

from docqa.config import Settings
from docqa.embeddings import create_embeddings
from docqa.graph import RagAgent, build_agent
from docqa.llm import create_chat_model
from docqa.reranker import create_reranker
from docqa.store import HybridStore
from docqa.tracing import get_tracing_handlers


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceRef(BaseModel):
    rank: int
    source: str
    chunk_index: int | None = None
    snippet: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


NO_CONTEXT_MESSAGE = "I could not find relevant information in the indexed documents."


def create_app(agent: RagAgent, settings: Settings, static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="docqa-rag", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        try:
            with agent.store.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "ready", "database": "up"}
        except Exception:
            return {"status": "not-ready", "database": "down"}

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        try:
            result = agent.ask(payload.question)
        except Exception as exc:  # pragma: no cover - surface DB/config errors
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        sources = [SourceRef(**s) for s in result["sources"]]
        return AskResponse(answer=result["answer"], sources=sources)

    @app.post("/ask/stream")
    def ask_stream(payload: AskRequest) -> StreamingResponse:
        config = {
            "configurable": {"thread_id": "default"},
            "callbacks": get_tracing_handlers(settings),
        }

        def event_stream():
            for update in agent.graph.stream(
                {"question": payload.question}, config=config, stream_mode="updates"
            ):
                if "retrieve" in update:
                    n = len(update["retrieve"].get("contexts", []))
                    yield _sse("status", {"node": "retrieve", "chunks": n})
                elif "generate" in update:
                    answer = update["generate"].get("answer", {}).get("answer", "")
                    yield _sse("answer", {"answer": answer})
                elif "no_context" in update:
                    yield _sse("answer", {"answer": NO_CONTEXT_MESSAGE})

            snapshot = agent.graph.get_state(config)
            contexts = snapshot.values.get("contexts", []) if snapshot is not None else []
            sources = [
                {
                    "rank": s.rank,
                    "source": s.document.metadata.get("source", "unknown"),
                    "chunk_index": s.document.metadata.get("chunk_index"),
                    "snippet": s.document.page_content[:240],
                }
                for s in contexts
            ]
            yield _sse("sources", {"sources": sources})
            yield _sse("done", {"ok": True})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    if static_dir is not None:
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return (static_dir / "index.html").read_text(encoding="utf-8")

    return app


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def get_app(settings: Settings | None = None) -> FastAPI:
    """Build the production app (DB + models wired). Import side-effect free."""
    settings = settings or Settings()
    store = HybridStore(
        settings.database_url,
        settings.collection_name,
        create_embeddings(settings),
    )
    agent = build_agent(
        store,
        create_chat_model(settings),
        create_reranker(settings),
        settings,
    )
    static_dir = Path(__file__).parent / "static"
    return create_app(agent, settings, static_dir=static_dir)
