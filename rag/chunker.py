# Author: Comfort Chiadi Esonu | Index: 10022200081

"""Manual chunking strategies for heterogeneous document corpora."""

from __future__ import annotations

import re
from statistics import mean
from typing import Dict, List


Chunk = Dict[str, str]


def _build_chunk(text: str, source: str, chunk_id: str, strategy: str) -> Chunk:
    """Create a standardized chunk dictionary object."""
    return {"text": text.strip(), "source": source, "chunk_id": chunk_id, "strategy": strategy}


def fixed_size_chunker(text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """Split text by character count using overlapping windows."""
    chunks: List[Chunk] = []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")
    start = 0
    counter = 0
    while start < len(text):
        end = start + chunk_size
        candidate = text[start:end].strip()
        if candidate:
            chunks.append(_build_chunk(candidate, source, f"{source}-fixed-{counter}", "fixed"))
            counter += 1
        start += chunk_size - overlap
    return chunks


def sentence_chunker(text: str, source: str, max_sentences: int = 5, overlap: int = 1) -> List[Chunk]:
    """Split text into chunks of sentences using regex boundaries."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks: List[Chunk] = []
    if max_sentences <= overlap:
        raise ValueError("max_sentences must be greater than overlap.")

    i = 0
    counter = 0
    step = max_sentences - overlap
    while i < len(sentences):
        current = " ".join(sentences[i : i + max_sentences]).strip()
        if current:
            chunks.append(_build_chunk(current, source, f"{source}-sentence-{counter}", "sentence"))
            counter += 1
        i += step
    return chunks


def paragraph_chunker(text: str, source: str) -> List[Chunk]:
    """Split text on paragraph boundaries and filter short chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[Chunk] = []
    counter = 0
    for paragraph in paragraphs:
        if len(paragraph) >= 100:
            chunks.append(_build_chunk(paragraph, source, f"{source}-paragraph-{counter}", "paragraph"))
            counter += 1
    return chunks


def compare_chunking_strategies(sample_text: str, source: str = "sample") -> None:
    """Run all chunkers on sample text and print descriptive statistics."""
    strategy_runs = {
        "fixed": fixed_size_chunker(sample_text, source),
        "sentence": sentence_chunker(sample_text, source),
        "paragraph": paragraph_chunker(sample_text, source),
    }
    print("Chunking comparison stats:")
    for name, chunks in strategy_runs.items():
        lengths = [len(chunk["text"]) for chunk in chunks]
        if lengths:
            print(
                f"- {name}: num_chunks={len(lengths)}, avg_len={mean(lengths):.2f}, "
                f"min_len={min(lengths)}, max_len={max(lengths)}"
            )
        else:
            print(f"- {name}: num_chunks=0, avg_len=0, min_len=0, max_len=0")


def chunk_corpus(texts: List[str], source_prefix: str, strategy: str) -> List[Chunk]:
    """Chunk a list of documents with a selected strategy."""
    all_chunks: List[Chunk] = []
    for idx, text in enumerate(texts):
        source = f"{source_prefix}_{idx}"
        if strategy == "fixed":
            all_chunks.extend(fixed_size_chunker(text, source))
        elif strategy == "sentence":
            all_chunks.extend(sentence_chunker(text, source))
        elif strategy == "paragraph":
            all_chunks.extend(paragraph_chunker(text, source))
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
    return all_chunks