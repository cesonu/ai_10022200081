# Author: Comfort Chiadi Esonu | Index: 10022200081

"""FAISS-based vector index with persistence and scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np


class VectorStore:
    """Store chunk embeddings in a FAISS index and serve similarity search."""

    def __init__(self) -> None:
        """Initialize empty vector store."""
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[Dict[str, str]] = []
        self.normalized_embeddings: np.ndarray | None = None

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings for cosine-similarity with inner product."""
        vectors = embeddings.astype("float32").copy()
        faiss.normalize_L2(vectors)
        return vectors

    def build_index(self, embeddings: np.ndarray, chunks: List[Dict[str, str]]) -> None:
        """Build normalized FAISS inner-product index from embeddings."""
        normalized = self._normalize(embeddings)
        self.index = faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)
        self.chunks = chunks
        self.normalized_embeddings = normalized

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, object]]:
        """Search top-k similar chunks and return ranked records."""
        if self.index is None:
            raise ValueError("Index has not been built or loaded.")

        query = self._normalize(query_embedding)
        scores, indices = self.index.search(query, top_k)
        results: List[Dict[str, object]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({"chunk": self.chunks[idx], "score": float(score), "rank": rank, "index": int(idx)})
        return results

    def save_index(self, path: str) -> None:
        """Persist FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index available to save.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target))

        chunks_path = target.with_suffix(".chunks.json")
        with chunks_path.open("w", encoding="utf-8") as handle:
            json.dump(self.chunks, handle, ensure_ascii=True, indent=2)

    def load_index(self, path: str) -> None:
        """Load FAISS index and associated chunk metadata from disk."""
        target = Path(path)
        chunks_path = target.with_suffix(".chunks.json")
        self.index = faiss.read_index(str(target))
        with chunks_path.open("r", encoding="utf-8") as handle:
            self.chunks = json.load(handle)
