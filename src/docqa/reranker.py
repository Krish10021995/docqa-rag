from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document


class Reranker:
    """Re-orders retrieved chunks. The 'none' implementation is a pass-through."""

    def rerank(
        self, query: str, docs: Sequence[Document], top_k: int
    ) -> list[tuple[Document, float]]:
        raise NotImplementedError


class PassThroughReranker(Reranker):
    def rerank(
        self, query: str, docs: Sequence[Document], top_k: int
    ) -> list[tuple[Document, float]]:
        n = min(top_k, len(docs))
        return [(docs[i], 1.0 / (i + 1)) for i in range(n)]


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker via sentence-transformers (needs the [local] extra)."""

    def __init__(self, model_name: str, top_k: int = 4) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - guarded for UX
            raise RuntimeError("Cross-encoder reranking needs the [local] extra") from exc

        self.model = CrossEncoder(model_name)
        self.default_top_k = top_k

    def rerank(
        self, query: str, docs: Sequence[Document], top_k: int
    ) -> list[tuple[Document, float]]:
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores, strict=True), key=lambda t: t[1], reverse=True)
        return [(d, float(s)) for d, s in ranked[:top_k]]


def create_reranker(settings) -> Reranker:
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker(settings.reranker_model)
    return PassThroughReranker()
