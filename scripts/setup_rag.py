"""
Setup RAG
=========
Indexes the Q&A biography data into ChromaDB for retrieval.

Usage:
    python scripts/setup_rag.py
"""

import sys
from pathlib import Path

# Add parent directory to path for config import
from artificial_isaac.rag.rag_helper import RAGHelper
from artificial_isaac.config import (
    CHROMA_DB_DIR,
    RAG_COLLECTION_NAME,
    RAG_DATA_FILE,
)


def main():
    print("=" * 60)
    print("RAG Setup — Indexing Q&A Data")
    print("=" * 60)
    print(f"\nData file: {RAG_DATA_FILE}")
    print(f"ChromaDB:  {CHROMA_DB_DIR}\n")

    rag_helper = RAGHelper(
        persist_directory=str(CHROMA_DB_DIR),
        collection_name=RAG_COLLECTION_NAME,
    )

    # Clear old data first
    rag_helper.clear_collection()

    try:
        count = rag_helper.index_qa_file(str(RAG_DATA_FILE))
        print(f"\n✓ Total indexed chunks: {count}")
        return True
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        print(f"  Run the RAG Survey tool (tools/rag_survey.html) to create your data.")
        return False
    except Exception as e:
        print(f"\n✗ Indexing failed: {e}")
        return False


if __name__ == "__main__":
    main()