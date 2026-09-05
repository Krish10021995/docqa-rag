from __future__ import annotations

import argparse
from pathlib import Path

from docqa.chunking import ingest
from docqa.config import get_settings
from docqa.embeddings import create_embeddings
from docqa.store import HybridStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Load documents into the pgvector hybrid store")
    parser.add_argument("data_dir", type=Path, help="Directory of .md/.txt documents")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--reset", action="store_true", help="Clear the collection first")
    parser.add_argument("--collection", default=None, help="Override collection name")
    args = parser.parse_args()

    settings = get_settings()
    if args.chunk_size:
        settings.chunk_size = args.chunk_size
    if args.chunk_overlap:
        settings.chunk_overlap = args.chunk_overlap
    if args.collection:
        settings.collection_name = args.collection

    store = HybridStore(
        database_url=settings.database_url,
        collection_name=settings.collection_name,
        embeddings=create_embeddings(settings),
    )
    count = ingest(
        args.data_dir, settings.chunk_size, settings.chunk_overlap, store, reset=args.reset
    )
    print(
        f"Ingested {count} chunks into collection '{settings.collection_name}' "
        f"(total now {store.count()})."
    )


if __name__ == "__main__":
    main()
