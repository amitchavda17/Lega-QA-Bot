from __future__ import annotations

import os
from typing import Dict, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

from utils.config import HYBRID_ALPHA, RERANK_FINAL_K, RERANK_TOP_K, TOP_K
from retrieval.bm25_index import BM25Index
from retrieval.chroma_store import ChromaStore

try:
    import logging

    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    from sentence_transformers import CrossEncoder

    _CROSS_ENCODER_AVAILABLE = True
except Exception:
    _CROSS_ENCODER_AVAILABLE = False


class HybridRetriever:
    """Hybrid retriever: Chroma (vector) + BM25, with optional cross-encoder re-ranking."""

    def __init__(self, store: ChromaStore, bm25_index: BM25Index, all_chunks: List[Dict]) -> None:
        self._store = store
        self._bm25 = bm25_index
        self._alpha = HYBRID_ALPHA
        self._cross_encoder = None
        self._chunk_lookup: Dict[str, Dict] = {c["chunk_id"]: c for c in all_chunks}

    def _get_cross_encoder(self):
        if not _CROSS_ENCODER_AVAILABLE:
            return None
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            import transformers

            transformers.logging.set_verbosity_error()
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self._cross_encoder = None
        return self._cross_encoder

    def retrieve(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        vector_query: Optional[str] = None,
        doc_id_filter: Optional[str] = None,
        rerank: bool = True,
    ) -> List[Dict]:
        """
        Hybrid search:
          - vector search in Chroma
          - BM25 over chunk texts
          - RRF-style score fusion
          - optional cross-encoder re-ranking
        """
        k = k or TOP_K
        recall_k = RERANK_TOP_K if rerank else k

        vec_q = vector_query or query
        where = {"doc_id": {"$eq": doc_id_filter}} if doc_id_filter else None
        vec_results = self._store.similarity_search(vec_q, k=recall_k * 2, where=where)

        vec_scores: Dict[str, float] = {}
        for rank, item in enumerate(vec_results):
            cid = item["chunk_id"]
            dist = item.get("distance")
            score = 1.0 / (1.0 + float(dist)) if dist is not None else 1.0 / (1.0 + rank)
            vec_scores[cid] = max(vec_scores.get(cid, 0.0), score)

        bm25_pairs = self._bm25.search(query, k=recall_k * 2)
        bm25_scores: Dict[str, float] = {cid: score for cid, score in bm25_pairs}

        _normalize(vec_scores)
        _normalize(bm25_scores)

        combined: Dict[str, float] = {}
        for cid in set(vec_scores) | set(bm25_scores):
            v = vec_scores.get(cid, 0.0)
            b = bm25_scores.get(cid, 0.0)
            combined[cid] = self._alpha * v + (1.0 - self._alpha) * b

        id_to_item: Dict[str, Dict] = {item["chunk_id"]: item for item in vec_results}

        scored = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        top_candidates: List[Dict] = []
        for cid, _ in scored[:recall_k]:
            if cid in id_to_item:
                top_candidates.append(id_to_item[cid])
            elif cid in self._chunk_lookup:
                c = self._chunk_lookup[cid]
                top_candidates.append({
                    "chunk_id": cid,
                    "text": c.get("text", ""),
                    "metadata": {
                        "doc_id": c.get("doc_id", ""),
                        "section_path": c.get("section_path", ""),
                        "section_title": c.get("section_title", ""),
                        "chunk_kind": c.get("chunk_kind", ""),
                    },
                    "distance": None,
                })

        if not rerank or not top_candidates:
            return top_candidates[:k]

        cross = self._get_cross_encoder()
        if not cross:
            return top_candidates[:k]

        pairs = [(query, c["text"]) for c in top_candidates]
        scores = cross.predict(pairs)
        order = np.argsort(scores)[::-1]
        final_k = min(k, RERANK_FINAL_K)
        return [top_candidates[int(idx)] for idx in order[:final_k]]


def _normalize(scores: Dict[str, float]) -> None:
    if not scores:
        return
    mx = max(scores.values())
    if mx > 0:
        for cid in scores:
            scores[cid] /= mx


def build_retriever(chunks: List[Dict]) -> HybridRetriever:
    """Build BM25 index + ChromaStore; Chroma must already be populated."""
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25Index(texts, ids)
    store = ChromaStore()
    return HybridRetriever(store, bm25, chunks)
