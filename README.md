# docqa-rag

Production RAG over policy documents with a **LangGraph** agent:
query rewriting, **hybrid retrieval** (pgvector ANN + Postgres full-text +
Reciprocal Rank Fusion), **optional cross-encoder reranking**, LLM generation
with **citations**, and **RAGAS** evaluation.

Built by [Krishnendu Pramanik](https://github.com/krish10021995).

## Pipeline

```
question
  └── rewrite            (query rewriting node, ready for LLM-based expansion)
  └── retrieve           (pgvector cosine + tsvector FTS, fused with RRF)
  └── rerank             (cross-encoder, optional)
  └── generate           (LLM answers with [CITATION:n] markers)
        └── no_context   (honest "I could not find relevant information" branch)
```

The graph is built with LangGraph (`src/docqa/graph.py`), state is a typed
`TypedDict`, and checkpoints run through LangGraph's `MemorySaver`. The result
is either answered with citations or routed to a no-context node — the agent
never guesses from an empty context.

## Stack

| Concern | Choice |
|---|---|
| Orchestration | LangGraph (`StateGraph`, conditional edges, checkpointing) |
| Vector store | pgvector on Postgres 16 (`langchain-postgres`) |
| Lexical search | Postgres full-text (`websearch_to_tsquery`, `ts_rank_cd`) |
| Fusion | Reciprocal Rank Fusion over the two ranked lists |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers, or any OpenAI-compatible endpoint |
| LLM | OpenAI-compatible (OpenAI / Groq / Ollama / vLLM) via `langchain-openai` |
| API | FastAPI, single endpoint + Server-Sent Events streaming |
| Tracing | Langfuse (optional extra) |
| Eval | RAGAS (optional extra) + a retrieval-only chunking sweep |

All models are behind thin factories so the system runs with **stub**
providers (no network, no keys) — CI and tests stay fast and deterministic.

## Quickstart (no keys, stub LLM)

```bash
uv sync                                 # install + lock
docker compose up -d db                 # pgvector on localhost:5433
uv run python -m docqa.ingestion data/policies --reset
uv run uvicorn docqa.server:app --port 8000 --reload
# open http://localhost:8000
```

The default embedder (`local`) downloads `BAAI/bge-small-en-v1.5` on first
use. To skip model downloads entirely:

```bash
DOCQA_EMBEDDING_PROVIDER=stub uv run uvicorn docqa.server:app --port 8000
```

## Docker

```bash
docker compose up --build            # db + api (+ run `docker compose run --rm ingest`)
```

Or one-shot in CI style:

```bash
docker compose run --rm ingest       # load data/policies into pgvector
curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"Can I work abroad for 6 weeks?"}'
```

## Configuration

Everything is `DOCQA_` prefixed and can live in `.env` (see
`src/docqa/config.py`).

| Variable | Default | Notes |
|---|---|---|
| `DOCQA_DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5433/docqa` | pgvector URL |
| `DOCQA_EMBEDDING_PROVIDER` | `local` | `local` \| `openai` \| `stub` |
| `DOCQA_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | or e.g. `nomic-embed-text` via Ollama |
| `DOCQA_LLM_PROVIDER` | `openai_compat` | `openai_compat` \| `stub` |
| `DOCQA_LLM_MODEL` | `gpt-4o-mini` | or `llama3.2` via Ollama/`DOCQA_OPENAI_API_BASE` |
| `DOCQA_OPENAI_API_BASE` | `` | any OpenAI-compatible base URL |
| `DOCQA_RERANKER` | `none` | `none` \| `cross-encoder` |
| `DOCQA_TOP_K` / `DOCQA_VECTOR_K` / `DOCQA_KEYWORD_K` | `4` / `8` / `8` | retrieval depths |
| `DOCQA_CHUNK_SIZE` / `DOCQA_CHUNK_OVERLAP` | `512` / `64` | ingestion |
| `DOCQA_LANGFUSE_PUBLIC_KEY` / `DOCQA_LANGFUSE_SECRET_KEY` | `` | enables Langfuse tracing |

## Evaluation

```bash
# RAGAS (needs OPENAI-compatible LLM credentials; uses the running store)
uv sync --extra eval
uv run python -m docqa.eval_ragas --eval-set data/eval/qa.jsonl --min-faithfulness 0.5

# retrieval-only chunking sweep (no LLM calls)
uv run python -m docqa.chunking_eval --grid 256,384,512,768,1024
```

See `docs/chunking-experiments.md` for the experiment write-up.

## Tests

```bash
uv run pytest -q            # unit tests (fast, no Postgres needed)
uv run pytest -q -m pg      # integration tests against a real pgvector instance
```

The `pg`-marked suite auto-skips when Postgres is unreachable, so local runs
stay green without Docker. GitHub Actions runs both suites against a
`pgvector/pgvector:pg16` service.

## Project layout

```
src/docqa/
  config.py        # DOCQA_ pydantic-settings
  embeddings.py    # openai / bge-small / stub factories
  store.py         # HybridStore: pgvector + FTS + Reciprocal Rank Fusion
  chunking.py      # load -> split -> ingest
  reranker.py      # pass-through / cross-encoder
  llm.py           # OpenAI-compatible / stub chat model
  graph.py         # LangGraph StateGraph: rewrite -> retrieve -> (rerank) -> generate|no_context
  api.py           # FastAPI app: /ask, /ask/stream (SSE), /health, /readyz
  server.py        # production entrypoint (import-safe, no side effects)
  tracing.py       # optional Langfuse handlers
  ingestion.py     # CLI: load data_dir into the store
  eval_ragas.py    # CLI: RAGAS evaluation over qa.jsonl
  chunking_eval.py # CLI: chunk-size sweep
data/
  policies/        # sample markdown documents
  eval/qa.jsonl    # questions + gold sources for the sweep
tests/             # unit + pg-marked integration tests
```

## Roadmap ideas

- Swap `MemorySaver` for a Postgres checkpointer to persist chat threads.
- Add LLM-based query rewriting and HyDE in the `rewrite` node.
- Add RAGAS gates to CI for merged-data regressions.

---

Status: **complete** — unit tests, pg integration tests, container build, and
the chunking experiment all run in GitHub Actions.