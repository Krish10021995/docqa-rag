from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

# langchain-postgres 0.0.17 schema: a shared `langchain_pg_embedding` rowset
# (column `document` + jsonb `cmetadata`) keyed to `langchain_pg_collection`
# by `collection_id`. We join on the collection name so FTS and count queries
# deliberately stay independent of the ORM-defined table-per-dim classes.
_COLLECTION_JOIN = """
FROM langchain_pg_embedding AS e
JOIN langchain_pg_collection AS c ON c.uuid = e.collection_id
WHERE c.name = :coll
"""


@dataclass
class ScoredChunk:
    document: Document
    score: float
    rank: int


class HybridStore:
    """pgvector (vector search) + Postgres full-text (BM25-style) with Reciprocal Rank Fusion."""

    def __init__(
        self,
        database_url: str,
        collection_name: str,
        embeddings,
        chunk_id_key: str = "chunk_id",
    ) -> None:
        self.conn_str = database_url
        self.collection = collection_name
        self.chunk_id_key = chunk_id_key
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.store = PGVector(
            embeddings=embeddings,
            connection=self.conn_str,
            collection_name=collection_name,
            use_jsonb=True,
        )

    def reset(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM langchain_pg_embedding AS e
                    USING langchain_pg_collection AS c
                    WHERE e.collection_id = c.uuid AND c.name = :coll
                    """
                ),
                {"coll": self.collection},
            )

    def add_documents(self, docs: Sequence[Document]) -> list[str]:
        keyed = []
        for d in docs:
            meta: dict[str, Any] = dict(d.metadata or {})
            chunk_id = hashlib.sha1(
                f"{meta.get('source', '')}:{meta.get('chunk_index', 0)}".encode()
            ).hexdigest()[:16]
            meta[self.chunk_id_key] = chunk_id
            keyed.append(Document(page_content=d.page_content, metadata=meta))
        return self.store.add_documents(keyed)

    def count(self) -> int:
        with self.engine.connect() as conn:
            sql = text(f"SELECT count(*) {_COLLECTION_JOIN}")
            return int(conn.execute(sql, {"coll": self.collection}).scalar_one())

    def _vector_hits(self, question: str, k: int) -> list[tuple[Document, float]]:
        if k <= 0:
            return []
        # distance 0..2 (cosine) -> similarity 0..1
        return [
            (doc, 1.0 - distance)
            for doc, distance in self.store.similarity_search_with_score(question, k=k)
        ]

    def _keyword_hits(self, question: str, k: int) -> list[tuple[Document, float]]:
        if k <= 0:
            return []
        sql = text(
            f"""
            SELECT e.document AS content, e.cmetadata AS metadata
            {_COLLECTION_JOIN}
            AND to_tsvector('english', e.document) @@ websearch_to_tsquery('english', :q)
            ORDER BY ts_rank_cd(
                to_tsvector('english', e.document),
                websearch_to_tsquery('english', :q)
            ) DESC
            LIMIT :k
            """
        )
        with self.engine.connect() as conn:
            rows = (
                conn.execute(sql, {"coll": self.collection, "q": question, "k": k}).mappings().all()
            )
        return [
            (
                Document(page_content=r["content"], metadata=r["metadata"]),
                float(r["metadata"].get("_score", 0.5)),
            )
            for r in rows
        ]

    @staticmethod
    def _rrf(ranked_lists: list[list[Document]], k: int = 60) -> dict[str, float]:
        scores: dict[str, float] = {}
        for lst in ranked_lists:
            for i, doc in enumerate(lst):
                key = doc.metadata.get("chunk_id") or doc.page_content
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + i + 1)
        return scores

    @staticmethod
    def _dedupe(docs: Sequence[Document]) -> list[Document]:
        seen: set[str] = set()
        out: list[Document] = []
        for d in docs:
            key = d.metadata.get("chunk_id") or d.page_content
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    def hybrid_search(
        self,
        question: str,
        vector_k: int = 8,
        keyword_k: int = 8,
        top_k: int = 4,
    ) -> list[ScoredChunk]:
        vector_hits = self._vector_hits(question, vector_k)
        keyword_hits = self._keyword_hits(question, keyword_k)

        fused: dict[str, float] = (
            self._rrf([[d for d, _ in vector_hits], [d for d, _ in keyword_hits]])
            if vector_hits or keyword_hits
            else {}
        )

        by_id: dict[str, tuple[Document, float]] = {}
        for d, s in vector_hits:
            key = d.metadata.get("chunk_id") or d.page_content
            by_id.setdefault(key, (d, s))
        for d, s in keyword_hits:
            key = d.metadata.get("chunk_id") or d.page_content
            by_id.setdefault(key, (d, s))

        ranked = sorted(
            ((by_id[k], score) for k, score in fused.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            ScoredChunk(document=doc, score=score, rank=i + 1)
            for i, ((doc, _), score) in enumerate(ranked[:top_k])
        ]
