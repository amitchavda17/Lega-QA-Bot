"""Entrypoint: load index, build retriever, start interactive REPL."""
from __future__ import annotations

from utils.config import DATA_DIR, CHILD_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP
from ingestion.chunking import chunk_all
from ingestion.loader import load_corpus
from retrieval.retriever import build_retriever
from pipeline import process_turn
from repl import run_loop


def main():
    print("Loading corpus and building retriever ...")
    corpus = load_corpus(DATA_DIR)
    chunks = chunk_all(corpus, child_max_chars=CHILD_MAX_CHARS, overlap_chars=CHUNK_OVERLAP, min_chars=CHUNK_MIN_CHARS)
    retriever = build_retriever(chunks)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    def handle(question, history, _extra):
        return process_turn(
            question,
            history=history,
            retriever=retriever,
            chunks=chunks,
            chunk_by_id=chunk_by_id,
        )

    run_loop(handle, retriever=retriever, chunks=chunks)


if __name__ == "__main__":
    main()
