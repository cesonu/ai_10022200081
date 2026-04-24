# Author: Comfort Chiadi Esonu | Index: 10022200081

"""Data loading and cleaning utilities for the RAG system."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    import pymupdf as fitz  # type: ignore


def _to_snake_case(name: str) -> str:
    """Convert a column name into snake_case format."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove non-ASCII artifacts from text."""
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_clean_csv(csv_path: str | Path) -> List[str]:
    """Load CSV file and return cleaned row-level text entries."""
    df = pd.read_csv(csv_path)
    df.columns = [_to_snake_case(col) for col in df.columns]
    df = df.dropna(how="all").drop_duplicates()

    for idx in range(df.shape[1]):
        column_series = df.iloc[:, idx]
        if str(column_series.dtype) == "object":
            df.iloc[:, idx] = column_series.astype(str).str.strip()

    row_corpus: List[str] = []
    for _, row in df.iterrows():
        parts = []
        for column, value in row.items():
            if pd.notna(value):
                text_value = str(value).strip()
                if text_value and text_value.lower() != "nan":
                    parts.append(f"{column.replace('_', ' ').title()}: {text_value}")
        if parts:
            row_corpus.append(" | ".join(parts))
    return row_corpus


def _is_header_or_footer(line: str) -> bool:
    """Heuristically detect recurring header or footer lines."""
    lowered = line.lower().strip()
    if not lowered:
        return True
    header_keywords = [
        "republic of ghana",
        "budget statement",
        "economic policy",
        "for the financial year",
        "ministry of finance",
        "page ",
    ]
    return any(keyword in lowered for keyword in header_keywords)


def load_and_clean_pdf(pdf_path: str | Path) -> List[str]:
    """Load PDF pages and return cleaned text per page."""
    document = fitz.open(pdf_path)
    pages: List[str] = []
    try:
        for page in document:
            text = page.get_text("text")
            lines = [line.strip() for line in text.splitlines()]
            filtered = [line for line in lines if not _is_header_or_footer(line)]
            cleaned = _clean_text(" ".join(filtered))
            if len(cleaned) > 80:
                pages.append(cleaned)
    finally:
        document.close()
    return pages


def load_clean_corpora(csv_path: str | Path, pdf_path: str | Path) -> Tuple[List[str], List[str]]:
    """Return cleaned corpora from the election CSV and budget PDF."""
    csv_rows = load_and_clean_csv(csv_path)
    pdf_pages = load_and_clean_pdf(pdf_path)
    return csv_rows, pdf_pages
