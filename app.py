# Author: Comfort Chiadi Esonu | Index: 10022200081

"""Streamlit entry point for the ACity custom RAG chatbot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

from rag.chunker import chunk_corpus
from rag.embedder import Embedder
from rag.generator import Generator
from rag.ingestion import load_clean_corpora
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever
from rag.vector_store import VectorStore

load_dotenv()

CSV_URL = "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
PDF_URL = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-andEconomic-Policy_v4.pdf"
PDF_FALLBACK_URL = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
CSV_PATH = Path("data/Ghana_Election_Result.csv")
PDF_PATH = Path("data/budget_2025.pdf")
INDEX_PATH = Path("data/faiss.index")


def download_if_missing(url: str, path: Path) -> None:
    """Download a dataset file only if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    candidate_urls = [url]
    if path.name == "budget_2025.pdf":
        candidate_urls.append(PDF_FALLBACK_URL)

    last_error = None
    for candidate_url in candidate_urls:
        try:
            response = requests.get(candidate_url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
            return
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(f"Failed to download dataset for {path.name}: {last_error}")


def build_chunks(csv_rows: List[str], pdf_pages: List[str]) -> List[Dict[str, str]]:
    """Create chunk objects using source-specific default strategies."""
    csv_chunks = chunk_corpus(csv_rows, source_prefix="csv_row", strategy="sentence")
    pdf_chunks = chunk_corpus(pdf_pages, source_prefix="pdf_page", strategy="paragraph")
    return csv_chunks + pdf_chunks


def read_feedback_stats(feedback_path: Path = Path("logs/feedback_log.jsonl")) -> Dict[str, int]:
    """Summarize helpful and not-helpful feedback counts."""
    helpful = 0
    not_helpful = 0
    if feedback_path.exists():
        for line in feedback_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if bool(payload.get("helpful")):
                helpful += 1
            else:
                not_helpful += 1
    return {"helpful": helpful, "not_helpful": not_helpful}


@st.cache_resource(show_spinner=False)
def initialize_system(force_rebuild: bool = False) -> Tuple[RAGPipeline, Retriever]:
    """Initialize or load the full RAG stack and print startup summary."""
    download_if_missing(CSV_URL, CSV_PATH)
    download_if_missing(PDF_URL, PDF_PATH)

    csv_rows, pdf_pages = load_clean_corpora(CSV_PATH, PDF_PATH)
    chunks = build_chunks(csv_rows, pdf_pages)
    embedder = Embedder(cache_path="data/embeddings.npy")
    vector_store = VectorStore()

    if INDEX_PATH.exists() and INDEX_PATH.with_suffix(".chunks.json").exists() and not force_rebuild:
        vector_store.load_index(str(INDEX_PATH))
    else:
        embeddings = embedder.embed_texts([chunk["text"] for chunk in chunks], use_cache=not force_rebuild)
        vector_store.build_index(embeddings=embeddings, chunks=chunks)
        vector_store.save_index(str(INDEX_PATH))

    retriever = Retriever(vector_store=vector_store, embedder=embedder, alpha=0.6)
    generator = Generator()
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    index_dimension = vector_store.index.d if vector_store.index is not None else 0
    print(
        "Startup summary:",
        f"chunks_indexed={len(vector_store.chunks)}",
        f"sources_loaded={2}",
        f"index_dimension={index_dimension}",
    )
    return pipeline, retriever


def main() -> None:
    """Render Streamlit UI and handle chat and feedback interactions."""
    st.set_page_config(page_title="ACity RAG Assistant", layout="wide")
    if "pipeline_obj" not in st.session_state:
        st.session_state.pipeline_obj = None
    if "retriever_obj" not in st.session_state:
        st.session_state.retriever_obj = None

    def ensure_initialized(force_rebuild: bool = False) -> bool:
        """Initialize pipeline objects on demand and store in session state."""
        try:
            with st.spinner("Initializing datasets, embeddings, and vector index..."):
                pipeline_obj, retriever_obj = initialize_system(force_rebuild=force_rebuild)
            st.session_state.pipeline_obj = pipeline_obj
            st.session_state.retriever_obj = retriever_obj
            return True
        except Exception as exc:
            st.error(f"Initialization failed: {exc}")
            return False

    st.sidebar.title("ACity RAG Assistant")
    template_display = st.sidebar.selectbox("Prompt Template", ["Strict", "Analytical", "Conversational"], index=0)
    top_k = st.sidebar.slider("Top-K results", min_value=1, max_value=10, value=5)
    alpha = st.sidebar.slider("Hybrid search alpha", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    feedback_stats = read_feedback_stats()
    st.sidebar.subheader("Feedback Stats")
    st.sidebar.write(f"Helpful: {feedback_stats['helpful']}")
    st.sidebar.write(f"Not Helpful: {feedback_stats['not_helpful']}")

    if st.sidebar.button("Rebuild Index"):
        st.cache_resource.clear()
        st.session_state.pipeline_obj = None
        st.session_state.retriever_obj = None
        if ensure_initialized(force_rebuild=True):
            st.success("Index rebuilt successfully.")

    if st.sidebar.button("Initialize System"):
        ensure_initialized(force_rebuild=False)

    with st.sidebar.expander("About this system"):
        st.write(
            "This assistant uses a custom RAG pipeline with manual ingestion, chunking, "
            "FAISS retrieval, hybrid ranking, prompt templates, and LLM generation."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    pipeline = st.session_state.pipeline_obj
    retriever = st.session_state.retriever_obj

    if pipeline is None or retriever is None:
        st.info("System is not initialized yet. Ask a question or click 'Initialize System' in the sidebar.")
    else:
        retriever.alpha = alpha

    query = st.chat_input("Ask a question about Ghana election results or the 2025 budget...")
    if query:
        if pipeline is None or retriever is None:
            if not ensure_initialized(force_rebuild=False):
                return
            pipeline = st.session_state.pipeline_obj
            retriever = st.session_state.retriever_obj
            retriever.alpha = alpha

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            try:
                result = pipeline.run(query=query, top_k=top_k, template=template_display.lower())
                st.write(result["llm_response"])
                st.session_state.messages.append({"role": "assistant", "content": result["llm_response"]})
                st.session_state.last_result = result

                with st.expander("Retrieved Chunks"):
                    table_rows = []
                    for item in result["retrieved_chunks"]:
                        table_rows.append(
                            {
                                "rank": item["rank"],
                                "source": item["source"],
                                "score": round(float(item["score"]), 4),
                                "text_preview": item["text"][:200],
                            }
                        )
                    st.table(table_rows)

                with st.expander("Prompt Sent to LLM"):
                    st.code(result["prompt_sent_to_llm"])

                with st.expander("Truncation Report"):
                    st.json(result["truncation_report"])

                timing = " | ".join(f"{stage['stage']}: {stage['ms']:.2f} ms" for stage in result["pipeline_log"])
                st.caption(f"Pipeline Timing: {timing}")

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")

    if st.session_state.last_result:
        last = st.session_state.last_result
        selected_chunks = last.get("selected_chunks", [])
        if selected_chunks:
            chunk_id = selected_chunks[0]["chunk"]["chunk_id"]
            col1, col2 = st.columns(2)
            if col1.button("thumbs_up", use_container_width=True):
                retriever.record_feedback(last["query"], chunk_id=chunk_id, helpful=True)
                st.success("Feedback recorded as helpful.")
            if col2.button("thumbs_down", use_container_width=True):
                retriever.record_feedback(last["query"], chunk_id=chunk_id, helpful=False)
                st.warning("Feedback recorded as not helpful.")


if __name__ == "__main__":
    main()