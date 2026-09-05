# Chunking experiments

`data/eval/qa.jsonl` contains questions with a `gold_source` file. The
chunking sweep answers: *which `chunk_size` gives the best retrieval
context hit-rate?*

## Run

```bash
uv sync --extra local         # once: local embeddings need the [local] extra
docker compose up -d db       # once: pgvector
uv run python -m docqa.ingestion data/policies --reset
uv run python -m docqa.chunking_eval --grid 256,384,512,768,1024
```

Each chunk size is ingested into its own collection (`policies_cs<size>`), then
`hybrid_search` (pgvector + FTS + RRF) is checked for `context_hit@4`.

## Baseline (sample documents)

| chunk_size | context_hit@4 |
|---|---|
| _(fill in after running)_ | (hit / questions) |

## What to expect

- Very small chunks (128-256) fragment answers across multiple chunks and can
  miss the gold source at `top_k=4`.
- Very large chunks (2048+) rarely hurt retrieval (the gold source is usually
  present) but degrade generation: more tokens per context, more noise.
- Middle ground (400-600, ~10% overlap) usually wins on the mix of
  faithfulness and precision; confirm with a RAGAS run.

## Method notes

- Retrieval-only metric: no LLM calls, fast and deterministic against the
  sample data.
- Use `uv run python -m docqa.eval_ragas --eval-set data/eval/qa.jsonl --min-faithfulness 0.5`
  afterwards to score the full pipeline end to end.