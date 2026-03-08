"""
vector_store.py
---------------
Pinecone vector store setup and retriever for the math agent.

Usage:
    from vector_store import get_retriever, upsert_documents

    # On first run / when adding new PDFs:
    from chunker import process_mathematical_documents
    docs = process_mathematical_documents("Data/")
    upsert_documents(docs)

    # Every other time (index already populated):
    retriever = get_retriever(k=3)
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from chunker import embeddings  # reuse — don't reload the model

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "math-agent"
DIMENSION        = 384          # matches all-MiniLM-L6-v2
METRIC           = "cosine"


# ──────────────────────────────────────────────
# Pinecone client (singleton)
# ──────────────────────────────────────────────

pc = Pinecone(api_key=PINECONE_API_KEY)


def _ensure_index_exists():
    """Create the Pinecone index if it doesn't exist yet."""
    if not pc.has_index(INDEX_NAME):
        print(f"Index '{INDEX_NAME}' not found. Creating...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index '{INDEX_NAME}' created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def get_vector_store() -> PineconeVectorStore:
    """Return a PineconeVectorStore connected to the existing index."""
    _ensure_index_exists()
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )


def get_retriever(k: int = 3):
    """
    Return a retriever ready to use in the RAG chain.

    Args:
        k: Number of documents to retrieve per query.
    """
    store = get_vector_store()
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def upsert_documents(docs, batch_size: int = 100):
    """
    Embed and upsert LangChain Documents into Pinecone.
    Call this from ingest.py — NOT at agent runtime.

    Args:
        docs:       List of LangChain Documents (output of chunker.py).
        batch_size: Upload in batches to avoid rate limits.
    """
    _ensure_index_exists()

    print(f"Upserting {len(docs)} documents in batches of {batch_size}...")
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=INDEX_NAME,
        )
        print(f"  Uploaded batch {i // batch_size + 1} / {-(-len(docs) // batch_size)}")

    print("Upsert complete.")