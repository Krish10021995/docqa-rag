from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = {".md", ".txt", ".rst", ".markdown"}


def load_documents(data_dir: Path) -> list[Document]:
    """Load all text documents under data_dir, labelled with their source filename."""
    docs: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        length_function=len,
        keep_separator=True,
    )
    chunks: list[Document] = []
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc.page_content)):
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = i
            metadata["n_chunks"] = i + 1
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks


def ingest(
    data_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    store,
    reset: bool = False,
) -> int:
    """Load -> chunk -> embed -> upsert. Returns number of chunks ingested."""
    if reset:
        store.reset()
    documents = split_documents(load_documents(data_dir), chunk_size, chunk_overlap)
    ids = store.add_documents(documents)
    return len(ids)
