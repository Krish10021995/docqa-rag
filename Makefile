.PHONY: install test lint format db-up db-down ingest serve eval chunk-eval clean

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

db-up:
	docker compose up -d db

db-down:
	docker compose down

ingest:
	uv run python -m docqa.ingestion data/policies --reset

serve:
	uv run uvicorn docqa.server:app --reload --port 8000

eval:
	uv run python -m docqa.eval_ragas --eval-set data/eval/qa.jsonl

chunk-eval:
	uv run python -m docqa.chunking_eval --grid 256,512,1024

clean:
	uv run ruff clean || true