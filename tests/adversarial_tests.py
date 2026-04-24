# Author: Comfort Chiadi Esonu | Index: 10022200081

"""Adversarial tests comparing RAG responses against direct LLM outputs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from rag.generator import Generator
from rag.pipeline import RAGPipeline


def _truncate(text: str, limit: int = 180) -> str:
    """Truncate long response text for markdown table display."""
    return (text[: limit - 3] + "...") if len(text) > limit else text


def run_adversarial_tests(pipeline: RAGPipeline, generator: Generator) -> None:
    """Execute adversarial test suite and append markdown comparison report."""
    tests: List[Tuple[str, str]] = [
        ("What happened in 2024?", "Should avoid unsupported broad claims and remain grounded."),
        ("Who won the Ghanaian presidential election in 2030?", "Should say no information for 2030."),
        ("What is the recipe for jollof rice?", "Should identify query as out of scope."),
        ("What percentage of the budget went to...", "Should request clarification or closest supported context."),
    ]

    lines = [
        "",
        "## Adversarial Evaluation",
        "",
        "| Query | RAG Response (truncated) | Pure LLM Response (truncated) | Hallucination Detected (Y/N) | Notes |",
        "|---|---|---|---|---|",
    ]

    for query, notes in tests:
        rag_result = pipeline.run(query=query, top_k=5, template="strict")
        rag_response = rag_result["llm_response"]
        pure_prompt = (
            "Answer this question directly without external context. "
            "If unsure, say you are unsure.\n\nQuestion: " + query
        )
        pure_response = generator.generate(pure_prompt)

        hallucination = "Y" if "2030" in query and "don't have enough information" not in rag_response.lower() else "N"
        lines.append(
            f"| {query} | {_truncate(rag_response)} | {_truncate(pure_response)} | {hallucination} | {notes} |"
        )

    output_path = Path("logs/adversarial_results.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    """Provide script entry guidance for adversarial tests."""
    raise RuntimeError("Initialize the pipeline in app context, then call run_adversarial_tests().")


if __name__ == "__main__":
    main()
