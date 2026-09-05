from __future__ import annotations

import argparse
import json
from pathlib import Path

from docqa.chunking import ingest
from docqa.config import Settings, get_settings


def load_eval_set(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep chunk sizes and report retrieval hit-rate (context_hit@k)"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/policies"))
    parser.add_argument("--eval-set", type=Path, default=Path("data/eval/qa.jsonl"))
    parser.add_argument(
        "--grid",
        default="256,384,512,768,1024",
        help="Comma-separated chunk sizes to sweep (each in its own collection)",
    )
    args = parser.parse_args()

    from docqa.embeddings import create_embeddings
    from docqa.store import HybridStore

    items = load_eval_set(args.eval_set)
    base = get_settings()

    print("| chunk_size | context_hit@4 |")
    print("|---|---|")

    for size in [int(s) for s in args.grid.split(",") if s.strip()]:
        settings = Settings(
            chunk_size=size,
            chunk_overlap=max(0, size // 8),
            collection_name=f"policies_cs{size}",
            **base.model_dump(exclude={"chunk_size", "chunk_overlap", "collection_name"}),
        )
        store = HybridStore(
            settings.database_url,
            settings.collection_name,
            create_embeddings(settings),
        )
        ingest(args.data_dir, size, settings.chunk_overlap, store, reset=True)

        hits = 0
        for item in items:
            chunk_ids = store.hybrid_search(item["question"], top_k=4, vector_k=8, keyword_k=8)
            sources = {c.document.metadata.get("source") for c in chunk_ids}
            gold = item.get("gold_source")
            if (gold and gold in sources) or not gold:
                hits += 1

        print(f"| {size} | {hits}/{len(items)} ({hits / len(items):.0%}) |")


if __name__ == "__main__":
    main()
