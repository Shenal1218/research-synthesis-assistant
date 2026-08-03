"""
rag_pipeline.py
================
RAG (Retrieval-Augmented Generation) pipeline for the Research Paper
Methodology & Synthesis Assistant.

Responsibilities:
  1. Ingest a folder of academic paper text/PDF files (20+ papers).
  2. Chunk the documents into overlapping passages.
  3. Embed chunks using a free HuggingFace sentence-transformer
     (all-MiniLM-L6-v2) -- no paid API needed for embeddings.
  4. Persist embeddings in a local Chroma vector database.
  5. Expose a `retrieve()` function used by the agents in agents.py.
  6. Provide a small retrieval-quality evaluation harness (5 queries).

Run this file directly to (re)build the vector store and print the
5-query evaluation report:

    python rag_pipeline.py --rebuild
"""

import os
import glob
import argparse
from typing import List, Dict, Optional

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "data", "papers")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4


def load_documents(papers_dir: str = PAPERS_DIR) -> List[Document]:
    """Load every .txt and .pdf file inside `papers_dir` as LangChain Documents.

    Each Document keeps `source` (filename) in its metadata so agents can
    cite papers by name later -- this is what lets Agent 1 write things
    like "(Source: attention_is_all_you_need.txt)".
    """
    documents: List[Document] = []

    txt_paths = glob.glob(os.path.join(papers_dir, "*.txt"))
    pdf_paths = glob.glob(os.path.join(papers_dir, "*.pdf"))

    if not txt_paths and not pdf_paths:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in {papers_dir}. "
            "Add at least 20 paper files before running the pipeline."
        )

    for path in txt_paths:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = os.path.basename(path)
        documents.extend(docs)

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = os.path.basename(path)
        documents.extend(docs)

    print(f"[rag_pipeline] Loaded {len(documents)} raw document(s) "
          f"from {len(txt_paths) + len(pdf_paths)} file(s).")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split loaded documents into overlapping chunks suitable for embedding.

    CHUNK_SIZE=800 / OVERLAP=120 is a deliberate choice: large enough to
    keep a methodology paragraph mostly intact, small enough that a single
    chunk doesn't dilute the embedding with unrelated content. Justify this
    trade-off in the viva using the evaluation report below.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[rag_pipeline] Split into {len(chunks)} chunk(s) "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the free, local HuggingFace embedding model.

    Using a local embedding model (rather than an API-based one) keeps the
    RAG pipeline free to run and avoids per-call embedding costs, which
    matters because embeddings are generated far more often than LLM calls
    (every chunk, at ingestion time).
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(rebuild: bool = False) -> Chroma:
    """Build (or load) the persistent Chroma vector store."""
    embeddings = get_embedding_model()

    if rebuild or not os.path.exists(PERSIST_DIR):
        documents = load_documents()
        chunks = chunk_documents(documents)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )
        print(f"[rag_pipeline] Vector store built and persisted at {PERSIST_DIR}.")
    else:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        print(f"[rag_pipeline] Loaded existing vector store from {PERSIST_DIR}.")

    return vector_store


_vector_store: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    """Lazy singleton accessor so agents.py / app.py don't rebuild repeatedly."""
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store(rebuild=False)
    return _vector_store


def retrieve(query: str, k: int = TOP_K) -> List[Dict]:
    """Retrieve the top-k most relevant chunks for a query.

    Returns a list of plain dicts (not raw LangChain objects) so this
    function's output can be dropped straight into the LangGraph state as
    JSON -- this dict-based contract IS the "Tool" that the Router/ReAct
    agent calls in agents.py::retrieval_node.
    """
    store = get_vector_store()
    results = store.similarity_search_with_relevance_scores(query, k=k)

    formatted = []
    for doc, score in results:
        formatted.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
            "content": doc.page_content,
            "relevance_score": round(float(score), 4),
        })
    return formatted


# --------------------------------------------------------------------------- #
# Retrieval quality evaluation (assignment requirement: 5 sample queries)
# --------------------------------------------------------------------------- #
EVALUATION_QUERIES = [
    "What evaluation metrics are used to measure model performance?",
    "What datasets were used for training and testing?",
    "What are the limitations mentioned by the authors?",
    "How is the proposed method different from prior baselines?",
    "What preprocessing steps were applied to the raw data?",
]


def run_evaluation(queries: Optional[List[str]] = None, k: int = TOP_K) -> None:
    """Print a human-readable retrieval-quality report for a set of queries.

    For each query we show: the query, the top-k sources retrieved, and
    their relevance scores. Paste this output (or a summarized table) into
    the README's RAG evaluation section, and use it as viva talking points
    to justify chunk_size/k choices.
    """
    queries = queries or EVALUATION_QUERIES
    print("\n" + "=" * 70)
    print("RAG RETRIEVAL QUALITY EVALUATION")
    print("=" * 70)

    for i, query in enumerate(queries, start=1):
        print(f"\nQuery {i}: {query}")
        print("-" * 70)
        results = retrieve(query, k=k)
        if not results:
            print("  No results retrieved.")
            continue
        for rank, r in enumerate(results, start=1):
            snippet = r["content"][:140].replace("\n", " ")
            print(f"  [{rank}] source={r['source']} "
                  f"score={r['relevance_score']} :: {snippet}...")

    print("\n" + "=" * 70)
    print("Manually inspect the above: for each query, do the top sources "
          "actually discuss what the query asks? Record a Precision@k "
          "estimate (e.g. 4/5 relevant) in the README for each query.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline builder & evaluator")
    parser.add_argument("--rebuild", action="store_true",
                         help="Force re-ingestion and re-embedding of all papers")
    args = parser.parse_args()

    build_vector_store(rebuild=args.rebuild)
    run_evaluation()
