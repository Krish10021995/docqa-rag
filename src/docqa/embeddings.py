from __future__ import annotations

from langchain_core.embeddings import Embeddings

from docqa.config import Settings


class StubEmbeddings(Embeddings):
    """Deterministic embeddings for tests and sandboxes when no model is available.

    Each token position produces a bounded, reproducible vector so retrieval
    behaviour stays stable across runs without downloading any model.
    """

    def __init__(self, dim: int = 16, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed

    def _vec(self, text: str) -> list[float]:
        start = sum(ord(c) for c in text) % self.seed
        return [float((start + i * 3) % 17) / 17.0 for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def create_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider.lower()

    if provider == "stub":
        return StubEmbeddings(dim=settings.embedding_dim)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key or "sk-none",
            openai_api_base=settings.openai_api_base,
            check_embedding_ctx_length=False,
        )

    if provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - guarded for UX
            raise RuntimeError(
                "Local embeddings need the [local] extra: `uv sync --extra local`"
            ) from exc

        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider}")


def embedding_dimension(settings: Settings, embeddings: Embeddings | None = None) -> int:
    probe = create_embeddings(settings) if embeddings is None else embeddings  # type: ignore[arg-type]
    vec = probe.embed_query("dimension probe")
    return len(vec)
