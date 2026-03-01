from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """Simple BM25 index over chunk texts."""

    def __init__(self, texts: Sequence[str], ids: Sequence[str]) -> None:
        self._ids = list(ids)
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        if not len(scores):
            return []
        idxs = np.argsort(scores)[::-1][:k]
        return [(self._ids[i], float(scores[i])) for i in idxs if scores[i] > 0.0]

