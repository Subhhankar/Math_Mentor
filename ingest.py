"""
ingest.py
---------
One-time script to chunk PDFs and populate the Pinecone index.

Run this ONCE when setting up, or whenever you add new PDF files.
Do NOT run this at agent startup — it's expensive and slow.

Usage:
    python ingest.py              # uses default "Data/" folder
    python ingest.py Data/        # explicit path
    python ingest.py Data/ --dry-run   # chunk only, skip Pinecone upload
"""

import sys
import argparse
from chunker import process_mathematical_documents
from vector_store import upsert_documents, pc, INDEX_NAME


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Pinecone.")
    parser.add_argument(
        "data_path",
        nargs="?",
        default="Data",
        help="Path to folder containing PDF files (default: Data/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk documents but skip Pinecone upload",
    )
    args = parser.parse_args()

    # ── Step 1: Chunk PDFs ───────────────────
    print(f"\n{'='*60}")
    print(f"INGEST PIPELINE")
    print(f"{'='*60}")
    print(f"Data path : {args.data_path}")
    print(f"Dry run   : {args.dry_run}")
    print(f"{'='*60}\n")

    docs = process_mathematical_documents(args.data_path)

    if not docs:
        print("No documents produced. Check your Data/ folder for PDFs.")
        sys.exit(1)

    # ── Step 2: Show summary ─────────────────
    topics = {}
    for doc in docs:
        t = doc.metadata.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1

    print(f"\nChunk summary:")
    print(f"  Total chunks : {len(docs)}")
    print(f"  By topic     :")
    for topic, count in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"    {topic:<20} {count} chunks")

    # ── Step 3: Upload to Pinecone ───────────
    if args.dry_run:
        print("\nDry run — skipping Pinecone upload.")
        return

    print(f"\nUploading to Pinecone index '{INDEX_NAME}'...")
    upsert_documents(docs)

    # ── Step 4: Verify ───────────────────────
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"\nPinecone index stats after ingest:")
    print(f"  Total vectors : {stats.total_vector_count}")
    print(f"\nIngest complete. Your knowledge base is ready.")


if __name__ == "__main__":
    main()