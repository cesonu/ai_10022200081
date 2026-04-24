# Academic City University RAG Chatbot

Student Name: [YOUR NAME]  
Index Number: [YOUR INDEX]

## Setup

1. Clone the repository.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Create a `.env` file and set:
   `ANTHROPIC_API_KEY=...`
   Optional fallback: `OPENAI_API_KEY=...`

## Run

`streamlit run app.py`

## Architecture Summary

This project implements a complete Retrieval-Augmented Generation pipeline manually, without LangChain, LlamaIndex, or other pre-built orchestration frameworks. It ingests a Ghana election CSV and a Ghana 2025 budget PDF, cleans them, and converts them into retrieval-ready chunked text.

The retrieval layer uses sentence-transformer embeddings with a FAISS cosine-similarity index for semantic search, then blends those scores with TF-IDF keyword scores to create a hybrid ranker. This supports both conceptual queries and exact-entity queries involving names, years, and numeric values.

The generation layer builds a constrained prompt from selected chunks and sends it to Anthropic Claude, with OpenAI fallback if needed. Every pipeline run logs retrieval scores, prompt content, truncation decisions, and stage timings for reproducible analysis and evaluation.

## Chunking Strategy Justification

The system includes fixed-size, sentence, and paragraph chunkers. A default 500-character chunk size with 50-character overlap is chosen because it preserves context continuity while preventing very large chunks that hurt retrieval precision. Sentence chunking is best for structured CSV rows, and paragraph chunking is best for budget narrative sections.

## Known Limitations

- PDF header/footer cleaning is heuristic and may keep or remove occasional lines imperfectly.
- Feedback re-ranking is lightweight and may need more data to produce strong effects.
- Query similarity for feedback boosting computes embeddings at runtime and can be optimized further.
- External LLM calls depend on API availability and quota.

## Links

- GitHub Repository: [ADD_GITHUB_REPO_URL](https://example.com)
- Deployed App: [ADD_DEPLOYED_APP_URL](https://example.com)
