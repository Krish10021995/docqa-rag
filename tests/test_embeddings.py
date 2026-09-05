from __future__ import annotations

from docqa.config import Settings
from docqa.embeddings import StubEmbeddings, create_embeddings, embedding_dimension


def test_stub_embeddings_deterministic():
    a = StubEmbeddings(dim=16)
    b = StubEmbeddings(dim=16)
    text = "what is the remote work policy?"
    assert a.embed_query(text) == b.embed_query(text)


def test_stub_embeddings_shape():
    emb = StubEmbeddings(dim=8)
    vecs = emb.embed_documents(["one", "two"])
    assert len(vecs) == 2
    assert all(len(v) == 8 for v in vecs)


def test_create_embeddings_stub():
    settings = Settings(embedding_provider="stub", embedding_dim=8)
    emb = create_embeddings(settings)
    assert isinstance(emb, StubEmbeddings)
    assert len(emb.embed_query("x")) == 8


def test_embedding_dimension_probes():
    settings = Settings(embedding_provider="stub", embedding_dim=12)
    assert embedding_dimension(settings) == 12
