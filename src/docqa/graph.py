from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from docqa.config import Settings
from docqa.reranker import Reranker
from docqa.store import HybridStore, ScoredChunk

SYSTEM_PROMPT = """You are a precise policy assistant.

Answer the question using ONLY the retrieved chunks below. Every claim in your
answer must be backed by one or more chunks, and you must cite them with
[CITATION:index] markers that match the context list.

If the chunks do not contain the answer, say so plainly and do not guess.

Contexts:
{contexts}
"""


class RagState(TypedDict, total=False):
    question: str
    rewritten_question: str
    contexts: list[ScoredChunk]
    answer: dict[str, Any]


class AnswerFrame(BaseModel):
    answer: str = Field(description="Markdown answer with [CITATION:n] markers")
    citations: list[int] = Field(description="Indices of contexts used")


def _serialize_contexts(contexts: list[ScoredChunk]) -> str:
    lines = []
    for item in contexts:
        chunk = item.document
        meta = chunk.metadata or {}
        label = f"[{item.rank}] ({meta.get('source', '?')}, chunk {meta.get('chunk_index', '?')})"
        lines.append(f"{label}\n{chunk.page_content}")
    return "\n\n".join(lines)


class RagAgent:
    def __init__(
        self, store: HybridStore, llm: BaseChatModel, reranker: Reranker, settings: Settings
    ):
        self.store = store
        self.llm = llm
        self.reranker = reranker
        self.settings = settings
        self.graph = self._build_graph()

    # ----- nodes -----

    def _rewrite(self, state: RagState) -> dict:
        return {"rewritten_question": state.get("question", "")}

    def _retrieve(self, state: RagState) -> dict:
        hits = self.store.hybrid_search(
            state["rewritten_question"],
            vector_k=self.settings.vector_k,
            keyword_k=self.settings.keyword_k,
            top_k=self.settings.top_k * 3,
        )
        if self.settings.reranker != "none":
            docs = [s.document for s in hits]
            reranked = self.reranker.rerank(state["rewritten_question"], docs, self.settings.top_k)
            hits = [
                ScoredChunk(document=doc, score=score, rank=i + 1)
                for i, (doc, score) in enumerate(reranked)
            ]
        else:
            hits = [
                ScoredChunk(document=s.document, score=s.score, rank=i + 1)
                for i, s in enumerate(hits[: self.settings.top_k])
            ]
        return {"contexts": hits}

    def _generate(self, state: RagState) -> dict:
        contexts = state.get("contexts", [])
        prompt = SYSTEM_PROMPT.format(contexts=_serialize_contexts(contexts))
        user = (
            f"Question: {state['rewritten_question']}\n\n"
            "Return a JSON object with keys 'answer' and 'citations'."
        )
        response = self.llm.invoke(
            [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
        )
        return {"answer": self._parse_answer(response.content)}

    def _no_context(self, state: RagState) -> dict:
        return {
            "answer": {
                "answer": "I could not find relevant information in the indexed documents.",
                "citations": [],
            }
        }

    # ----- helpers -----

    def _parse_answer(self, content: Any) -> dict[str, Any]:
        text = str(content).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return {"answer": text, "citations": []}
        try:
            frame = AnswerFrame.model_validate_json(text[start : end + 1])
            return {"answer": frame.answer, "citations": frame.citations}
        except Exception:
            return {"answer": text, "citations": []}

    def sources(self, state: RagState) -> list[dict]:
        citations = set(state.get("answer", {}).get("citations", []))
        return [
            {
                "rank": s.rank,
                "source": s.document.metadata.get("source", "unknown"),
                "chunk_index": s.document.metadata.get("chunk_index"),
                "snippet": s.document.page_content[:240],
            }
            for s in state.get("contexts", [])
            if s.rank in citations
        ]

    # ----- graph -----

    def _build_graph(self):
        builder = StateGraph(RagState)

        builder.add_node("rewrite", self._rewrite)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("no_context", self._no_context)

        builder.add_edge(START, "rewrite")
        builder.add_edge("rewrite", "retrieve")

        def _route(state: RagState) -> str:
            return "generate" if state.get("contexts") else "no_context"

        builder.add_conditional_edges(
            "retrieve", _route, {"generate": "generate", "no_context": "no_context"}
        )
        builder.add_edge("generate", END)
        builder.add_edge("no_context", END)

        return builder.compile(checkpointer=MemorySaver())

    def ask(self, question: str) -> dict:
        config = {"configurable": {"thread_id": "default"}}
        state = self.graph.invoke({"question": question}, config=config)
        answer = state["answer"]
        return {
            "answer": answer["answer"],
            "sources": self.sources(state),
        }

    def stream(self, question: str):
        config = {"configurable": {"thread_id": "default"}}
        yield from self.graph.stream({"question": question}, config=config, stream_mode="updates")


def build_agent(
    store: HybridStore, llm: BaseChatModel, reranker: Reranker, settings: Settings
) -> RagAgent:
    return RagAgent(store=store, llm=llm, reranker=reranker, settings=settings)
