# Author: Comfort Chiadi Esonu | Index: 10022200081

"""End-to-end RAG pipeline orchestration and logging."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from rag.generator import Generator
from rag.prompt_builder import build_prompt, context_window_manager
from rag.retriever import Retriever


class RAGPipeline:
    """Coordinate retrieval, prompt creation, generation, and structured output."""

    def __init__(self, retriever: Retriever, generator: Generator, log_path: str = "logs/pipeline_log.jsonl") -> None:
        """Initialize pipeline dependencies and run-log path."""
        self.retriever = retriever
        self.generator = generator
        self.log_path = Path(log_path)

    def _append_log(self, payload: Dict[str, object]) -> None:
        """Append a pipeline run payload to JSONL file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _build_llm_error_fallback(self, selected: List[Dict[str, object]]) -> str:
        """Return a safe user-facing response when LLM providers fail."""
        context_lines = []
        for item in selected[:3]:
            source = str(item["chunk"]["source"])
            preview = str(item["chunk"]["text"])[:220].strip()
            context_lines.append(f"- Source: {source} | Preview: {preview}")
        context_block = "\n".join(context_lines) if context_lines else "- No context chunks available."
        return (
            "LLM generation is temporarily unavailable because API quota/rate limits were reached.\n"
            "Retrieval still worked, and the most relevant document snippets are below:\n"
            f"{context_block}\n"
            "Please add billing/credits to GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY and try again."
        )

    def run(self, query: str, top_k: int = 5, template: str = "strict") -> Dict[str, object]:
        """Execute full RAG flow and return detailed structured result."""
        stages: List[Dict[str, float]] = []

        print("Stage 1/5: Retrieval")
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(query=query, top_k=top_k)
        stages.append({"stage": "retrieval", "ms": (time.perf_counter() - t0) * 1000})

        print("Stage 2/5: Context selection")
        t1 = time.perf_counter()
        selected, truncation_report = context_window_manager(retrieved, max_chars=3000)
        stages.append({"stage": "context_selection", "ms": (time.perf_counter() - t1) * 1000})

        print("Stage 3/5: Prompt construction")
        t2 = time.perf_counter()
        prompt = build_prompt(query=query, chunks=selected, template_name=template)
        stages.append({"stage": "prompt_build", "ms": (time.perf_counter() - t2) * 1000})

        print("Stage 4/5: LLM call")
        t3 = time.perf_counter()
        try:
            response = self.generator.generate(prompt)
        except Exception as exc:
            error_text = str(exc).lower()
            quota_markers = [
                "429",
                "insufficient_quota",
                "rate limit",
                "quota",
                "credit balance",
                "no api key found",
                "api key",
                "authentication",
            ]
            if any(marker in error_text for marker in quota_markers):
                response = self._build_llm_error_fallback(selected)
            else:
                raise
        stages.append({"stage": "llm_call", "ms": (time.perf_counter() - t3) * 1000})

        print("Stage 5/5: Response packaging")
        t4 = time.perf_counter()
        output = {
            "query": query,
            "retrieved_chunks": [
                {
                    "text": item["chunk"]["text"],
                    "source": item["chunk"]["source"],
                    "score": item["score"],
                    "rank": item["rank"],
                    "chunk_id": item["chunk"]["chunk_id"],
                }
                for item in retrieved
            ],
            "selected_chunks": selected,
            "truncation_report": truncation_report,
            "prompt_sent_to_llm": prompt,
            "llm_response": response,
            "template_used": template,
            "retrieval_scores": [float(item["score"]) for item in retrieved],
            "pipeline_log": stages,
        }
        stages.append({"stage": "response_packaging", "ms": (time.perf_counter() - t4) * 1000})
        output["pipeline_log"] = stages

        self._append_log({"timestamp": datetime.utcnow().isoformat(), **output})
        return output
