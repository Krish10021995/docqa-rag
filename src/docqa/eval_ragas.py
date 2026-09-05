from __future__ import annotations

import argparse
import json
from pathlib import Path

from docqa.config import get_settings
from docqa.embeddings import create_embeddings
from docqa.graph import build_agent
from docqa.llm import create_chat_model
from docqa.reranker import create_reranker
from docqa.store import HybridStore


def load_eval_set(path: Path) -> list[dict]:
    questions = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation over the eval set")
    parser.add_argument("--eval-set", type=Path, default=Path("data/eval/qa.jsonl"))
    parser.add_argument("--min-faithfulness", type=float, default=0.0)
    args = parser.parse_args()

    settings = get_settings()
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

    questions = load_eval_set(args.eval_set)
    print(f"Evaluating {len(questions)} questions with RAGAS...")

    rows = []
    for item in questions:
        result = agent.ask(item["question"])
        rows.append(
            {
                "user_input": item["question"],
                "response": result["answer"],
                "retrieved_contexts": [s["snippet"] for s in result["sources"]],
                "reference": item.get("ground_truth", ""),
            }
        )

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("RAGAS evaluation needs the [eval] extra: `uv sync --extra eval`") from exc

    ds = Dataset.from_list(rows)
    score = evaluate(
        ds,
        metrics=[answer_relevancy, context_precision, context_recall, faithfulness],
    )
    frame = score.to_pandas().round(3)
    print(frame.to_markdown(index=False))

    faithfulness = float(score["faithfulness"])
    if faithfulness < args.min_faithfulness:
        raise SystemExit(f"Faithfulness {faithfulness:.3f} below gate {args.min_faithfulness:.3f}")
    print(f"gate OK: faithfulness {faithfulness:.3f} >= {args.min_faithfulness:.3f}")


if __name__ == "__main__":
    main()
