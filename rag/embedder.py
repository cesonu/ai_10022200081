# Author: Comfort Chiadi Esonu | Index: 10022200081

"""Embedding pipeline built on sentence-transformers."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Generate and cache dense embeddings for chunks and queries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_path: str = "data/embeddings.npy") -> None:
        """Initialize embedding model and cache path."""
        self.model = SentenceTransformer(model_name)
        self.cache_path = Path(cache_path)

    def embed_texts(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Embed a list of texts, loading/saving cache when enabled."""
        if use_cache and self.cache_path.exists():
            return np.load(self.cache_path)
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        if use_cache:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, embeddings)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query and return a 2D vector array."""
        vector = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        return vector.astype("float32")