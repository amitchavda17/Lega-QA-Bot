from __future__ import annotations

from typing import Dict, List, Optional

import chromadb
from langchain_ollama import OllamaEmbeddings

from utils.config import CHROMA_PATH, EMBED_MODEL


class ChromaStore:
    """Thin wrapper around a Chroma collection and Ollama embeddings."""

    def __init__(self, collection_name: str = "contracts") -> None:
        self._client = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection = self._client.get_or_create_collection(name=collection_name)
        self._embedder = OllamaEmbeddings(model=EMBED_MODEL)

    def similarity_search(
        self,
        query_text: str,
        k: int,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Run vector similarity search in Chroma using an Ollama embedding of query_text.

        Returns list of dicts with: chunk_id, text, metadata, distance.
        """
        embedding = self._embedder.embed_query(query_text)
        # Chroma expects where to be omitted when no filters are used.
        where_clause = where if where else None
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where_clause,
            include=["metadatas", "documents", "distances"],
        )

        metadatas = res.get("metadatas", [[]])[0]
        docs = res.get("documents", [[]])[0]
        ids = [m.get("chunk_id") for m in metadatas]
        distances = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(ids)

        out: List[Dict] = []
        for i, cid in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            out.append(
                {
                    "chunk_id": cid,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": meta,
                    "distance": distances[i] if i < len(distances) else None,
                }
            )
        return out

