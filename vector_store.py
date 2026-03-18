"""
vector_store.py
---------------
Pinecone vector store setup and retriever for the math agent.
"""

import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from chunker import embeddings

INDEX_NAME = "math-agent"
DIMENSION  = 384
METRIC     = "cosine"


def _get_pc():
    """Get Pinecone client — reads API key at call time, not import time."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set. Add it to Streamlit secrets or .env file.")
    return Pinecone(api_key=api_key)


def _ensure_index_exists(pc):
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def get_vector_store() -> PineconeVectorStore:
    pc = _get_pc()
    _ensure_index_exists(pc)
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )


def get_retriever(k: int = 3):
    store = get_vector_store()
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def upsert_documents(docs, batch_size: int = 100):
    pc = _get_pc()
    _ensure_index_exists(pc)
    for i in range(0, len(docs), batch_size):
        batch = docs[i: i + batch_size]
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=INDEX_NAME,
        )
