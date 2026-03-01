"""
Offline index builder.

Usage:
  python build_index.py

Full rebuild: load docs → chunk → embed → store in Chroma.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import chromadb
from langchain_ollama import OllamaEmbeddings

from utils.config import (
    CHROMA_PATH,
    CHILD_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP,
    DATA_DIR,
    EMBED_MODEL,
)
from ingestion.loader import load_corpus
from ingestion.chunking import chunk_all


def build_index() -> None:
    """Full rebuild of the Chroma index from DATA_DIR."""
    print(f"[build] data_dir = {DATA_DIR}")
    print(f"[build] chroma   = {CHROMA_PATH}")

    docs = load_corpus(DATA_DIR)
    print(f"[build] loaded {len(docs)} documents: {[d[1] for d in docs]}")

    chunks: List[Dict] = chunk_all(
        docs,
        child_max_chars=CHILD_MAX_CHARS,
        overlap_chars=CHUNK_OVERLAP,
        min_chars=CHUNK_MIN_CHARS,
    )
    if not chunks:
        print("[build] No chunks produced; nothing to index.")
        return

    counts = Counter(c["doc_id"] for c in chunks)
    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
    print(f"[build] {len(chunks)} chunks total (avg {avg_len:.0f} chars)")
    for doc_id, n in sorted(counts.items()):
        print(f"  {doc_id}: {n} chunks")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(name="contracts")
        print("[build] dropped existing collection")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="contracts")

    embedder = OllamaEmbeddings(model=EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "doc_title": c.get("doc_title", ""),
            "section_path": c.get("section_path", ""),
            "section_title": c.get("section_title", ""),
            "chunk_kind": c.get("chunk_kind", ""),
        }
        for c in chunks
    ]
    ids = [c["chunk_id"] for c in chunks]

    BATCH = 50
    total = len(texts)
    for i in range(0, total, BATCH):
        batch_texts = texts[i : i + BATCH]
        batch_meta = metadatas[i : i + BATCH]
        batch_ids = ids[i : i + BATCH]
        vectors = embedder.embed_documents(batch_texts)
        collection.add(ids=batch_ids, embeddings=vectors, metadatas=batch_meta, documents=batch_texts)
        print(f"[build] embedded {min(i + BATCH, total)}/{total}")

    print("[build] done")


if __name__ == "__main__":
    build_index()
