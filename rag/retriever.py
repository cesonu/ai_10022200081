# Author: Comfort Chiadi Esonu | Index: 10022200081

"""Retriever implementation with hybrid search and feedback re-ranking."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.embedder import Embedder
from rag.vector_store import VectorStore


class Retriever:
    """Retrieve chunks using dense vectors and TF-IDF keyword matching."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        alpha: float = 0.6,
        retrieval_log_path: str = "logs/retrieval_log.jsonl",
        feedback_log_path: str = "logs/feedback_log.jsonl",
    ) -> None:
        """Initialize retriever with dense index, keyword index, and logs."""
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha
        self.retrieval_log_path = Path(retrieval_log_path)
        self.feedback_log_path = Path(feedback_log_path)
        self.last_results: Dict[str, List[Dict[str, object]]] = {}

        corpus = [chunk["text"] for chunk in self.vector_store.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def _append_jsonl(self, path: Path, payload: Dict[str, object]) -> None:
        """Append a dictionary payload to a JSONL log file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _keyword_scores(self, query: str) -> np.ndarray:
        """Calculate normalized TF-IDF cosine similarity scores for query."""
        if self.tfidf_matrix is None:
            return np.zeros(len(self.vector_store.chunks), dtype=float)
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        """Return top-k hybrid-ranked retrieval results for a query."""
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k=max(top_k * 3, top_k))
        keyword_scores = self._keyword_scores(query)

        rescored: List[Dict[str, object]] = []
        for result in vector_results:
            idx = result["index"]
            vector_score = float(result["score"])
            keyword_score = float(keyword_scores[idx]) if idx < len(keyword_scores) else 0.0
            final_score = (self.alpha * vector_score) + ((1 - self.alpha) * keyword_score)
            rescored.append(
                {
                    "chunk": result["chunk"],
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "score": final_score,
                    "index": idx,
                }
            )

        rescored.sort(key=lambda item: item["score"], reverse=True)
        ranked = []
        for rank, item in enumerate(rescored[:top_k], start=1):
            item["rank"] = rank
            ranked.append(item)

        self.last_results[query] = ranked
        for item in ranked:
            self._append_jsonl(
                self.retrieval_log_path,
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "query": query,
                    "chunk_id": item["chunk"]["chunk_id"],
                    "source": item["chunk"]["source"],
                    "rank": item["rank"],
                    "vector_score": item["vector_score"],
                    "keyword_score": item["keyword_score"],
                    "final_score": item["score"],
                },
            )
        return ranked

    def record_feedback(self, query: str, chunk_id: str, helpful: bool) -> None:
        """Store user feedback for a retrieved chunk in append-only log."""
        retrieved = self.last_results.get(query, [])
        score_at_retrieval = None
        for item in retrieved:
            if item["chunk"]["chunk_id"] == chunk_id:
                score_at_retrieval = item["score"]
                break
        self._append_jsonl(
            self.feedback_log_path,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "chunk_id": chunk_id,
                "helpful": helpful,
                "score_at_retrieval": score_at_retrieval,
            },
        )

    def _read_feedback(self) -> List[Dict[str, object]]:
        """Load feedback log entries if available."""
        if not self.feedback_log_path.exists():
            return []
        rows: List[Dict[str, object]] = []
        with self.feedback_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def get_feedback_boosted_results(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        """Retrieve results and re-rank using feedback from similar queries."""
        base_results = self.retrieve(query, top_k=top_k)
        feedback_entries = self._read_feedback()
        if not feedback_entries:
            return base_results

        query_embedding = self.embedder.embed_query(query)
        for result in base_results:
            adjustment = 0.0
            for entry in feedback_entries:
                past_query = entry.get("query", "")
                past_embedding = self.embedder.embed_query(str(past_query))
                similarity = float(cosine_similarity(query_embedding, past_embedding)[0][0])
                if similarity > 0.8 and entry.get("chunk_id") == result["chunk"]["chunk_id"]:
                    adjustment += 0.1 if bool(entry.get("helpful")) else -0.1
            result["score"] += adjustment
            result["feedback_adjustment"] = adjustment

        base_results.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(base_results, start=1):
            item["rank"] = rank
        return base_results[:top_k]

    def demonstrate_failure_case(self) -> None:
        """Compare pure vector retrieval against hybrid retrieval on vague query."""
        query = "Tell me what changed dramatically in the national situation."
        original_alpha = self.alpha
        self.alpha = 1.0
        vector_only = self.retrieve(query, top_k=5)
        self.alpha = original_alpha
        hybrid = self.retrieve(query, top_k=5)

        log_path = Path("logs/failure_case_log.md")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Failure Case - {datetime.utcnow().isoformat()}\n")
            handle.write(f"Query: {query}\n\n")
            handle.write("### Pure Vector Results\n")
            for item in vector_only:
                handle.write(f"- Rank {item['rank']} | Score {item['score']:.4f} | {item['chunk']['source']}\n")
            handle.write("\n### Hybrid Results\n")
            for item in hybrid:
                handle.write(f"- Rank {item['rank']} | Score {item['score']:.4f} | {item['chunk']['source']}\n")