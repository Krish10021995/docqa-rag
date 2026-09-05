FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[local]"

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src ./src
COPY data ./data

ENV PYTHONUNBUFFERED=1
ENV DOCQA_DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/docqa"

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "uvicorn", "docqa.server:app", "--host", "0.0.0.0", "--port", "8000"]