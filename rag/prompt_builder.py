# Author: Comfort Chiadi Esonu | Index: 10022200081

"""Prompt templating and context-window management utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple


TEMPLATE_STRICT = """You are the Academic City University RAG assistant.
Use only the context below to answer.
If the answer is not in context, respond exactly:
"I don't have enough information in the provided documents to answer this."
Never fabricate facts.

Context:
{context}

Sources:
{source_list}

Question:
{query}
"""

TEMPLATE_ANALYTICAL = """You are the Academic City University RAG assistant.
Analyze the retrieved context and synthesize a careful response.
If evidence is partial, explicitly acknowledge uncertainty.

Context:
{context}

Sources:
{source_list}

Question:
{query}
"""

TEMPLATE_CONVERSATIONAL = """You are the Academic City University RAG assistant.
Answer in a natural and clear tone while staying grounded in context.
You may make light inferences when clearly supported by evidence.

Context:
{context}

Sources:
{source_list}

Question:
{query}
"""


def context_window_manager(chunks: List[Dict[str, object]], max_chars: int = 3000) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Select highest-scoring chunks within a character budget."""
    ordered = sorted(chunks, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    selected: List[Dict[str, object]] = []
    consumed = 0
    dropped = 0
    for chunk in ordered:
        text = str(chunk["chunk"]["text"])
        if consumed + len(text) > max_chars:
            dropped += 1
            continue
        selected.append(chunk)
        consumed += len(text)
    report = {
        "max_chars": max_chars,
        "used_chars": consumed,
        "selected_count": len(selected),
        "dropped_count": dropped,
    }
    return selected, report


def build_prompt(query: str, chunks: List[Dict[str, object]], template_name: str = "strict") -> str:
    """Build a prompt string from query and selected chunk objects."""
    context = "\n\n".join(chunk["chunk"]["text"] for chunk in chunks)
    source_list = ", ".join(sorted({chunk["chunk"]["source"] for chunk in chunks}))

    template_map = {
        "strict": TEMPLATE_STRICT,
        "analytical": TEMPLATE_ANALYTICAL,
        "conversational": TEMPLATE_CONVERSATIONAL,
    }
    template = template_map.get(template_name.lower(), TEMPLATE_STRICT)
    return template.format(context=context, query=query, source_list=source_list)
