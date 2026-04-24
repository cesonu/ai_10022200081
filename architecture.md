# RAG Architecture

## Data Flow

```text
[Raw Data] -> [Ingestion & Cleaning] -> [Chunker] -> [Embedder] -> [FAISS Index]
                                                                  |
[User Query] -> [Query Embedder] -> [Retriever (Hybrid)] -> [Context Window Mgr]
                                                                  |
                                         [Prompt Builder] -> [Claude API] -> [Response UI]
```

## Component Rationale

- `ingestion.py` uses `pandas` and `PyMuPDF` because both are reliable for structured CSV rows and page-level PDF extraction.
- `chunker.py` provides fixed-size, sentence-based, and paragraph-based chunking so the system can handle both tabular election rows and narrative budget text.
- `embedder.py` uses `all-MiniLM-L6-v2`, which is lightweight and performant on local hardware.
- `vector_store.py` uses FAISS for fast cosine similarity retrieval at scale.
- `retriever.py` combines dense semantic retrieval with TF-IDF keyword matching, then applies optional user-feedback re-ranking.
- `prompt_builder.py` enforces context constraints through template selection and context-window control.
- `generator.py` uses Anthropic Claude first, then OpenAI fallback, ensuring continuity.
- `pipeline.py` centralizes orchestration, timings, and append-only experiment logs.

## Chunk Size Justification

The default 500-character chunks with 50-character overlap balance context richness against retrieval precision. CSV election records are short and specific, while budget paragraphs are longer and discursive; this window size captures enough neighboring context to preserve meaning without over-broad retrieval matches.

## Why Hybrid Search Fits This Domain

Election and budget questions often include exact entities, constituency names, and numeric values. Dense vectors capture semantic similarity, but keyword scoring helps preserve precision for exact tokens and figures. A weighted BM25-style TF-IDF component improves ranking robustness for factual, number-heavy academic queries.
