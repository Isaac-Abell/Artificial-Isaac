"""
RAG Helper — Simplified Q&A Format
====================================
Indexes and queries personal knowledge stored as flat Q&A JSON.

Expected JSON format:
[
  {"question": "Where did you grow up?", "answer": "Seattle, WA."},
  ...
]
"""

import chromadb
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any


class RAGHelper:
    """
    Helper class for indexing and querying Q&A knowledge in ChromaDB.
    Designed for the simplified flat JSON format: [{"question": ..., "answer": ...}]
    """

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "personal_rag"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _generate_id(source: str, index: int) -> str:
        """Generate a stable unique ID for a Q&A chunk."""
        content = f"{source}::qa_{index}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def load_qa_json(file_path: str) -> List[Dict[str, str]]:
        """
        Load and validate Q&A JSON from a file.

        Args:
            file_path: Path to a JSON file with [{"question": ..., "answer": ...}] format

        Returns:
            List of dicts with 'question' and 'answer' keys (only non-empty answers)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the JSON structure is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"RAG data file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

        qa_pairs = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dict: {type(item).__name__}")
            if "question" not in item or "answer" not in item:
                raise ValueError(f"Item {i} missing 'question' or 'answer' key")

            answer = str(item["answer"]).strip()
            if answer:  # Skip empty answers
                qa_pairs.append({
                    "question": str(item["question"]).strip(),
                    "answer": answer
                })

        return qa_pairs

    def index_qa_file(self, file_path: str) -> int:
        """
        Load a Q&A JSON file and index all pairs into ChromaDB.

        Each Q&A pair becomes a single document: "Q: {question}\nA: {answer}"

        Args:
            file_path: Path to the Q&A JSON file

        Returns:
            Number of chunks indexed
        """
        qa_pairs = self.load_qa_json(file_path)
        file_name = Path(file_path).stem

        if not qa_pairs:
            print("⚠ No Q&A pairs with answers found.")
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, pair in enumerate(qa_pairs):
            doc_id = self._generate_id(file_name, i)
            text = f"Q: {pair['question']}\nA: {pair['answer']}"

            ids.append(doc_id)
            documents.append(text)
            metadatas.append({
                "source_file": file_name,
                "question": pair["question"],
                "index": i,
                "chunk_type": "qa_pair"
            })

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"✓ Indexed {len(ids)} Q&A pairs from {file_name}.json")
        return len(ids)

    def query_context(self, prompt: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the RAG system for the most relevant context.
        """
        # Don't query more results than exist in the collection
        count = self.collection.count()
        if count == 0:
            return []
        n = min(n_results, count)

        results = self.collection.query(
            query_texts=[prompt],
            n_results=n
        )

        contexts = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0] if "distances" in results else [None] * len(ids_list)

        for i in range(len(ids_list)):
            contexts.append({
                "id": ids_list[i],
                "text": docs_list[i],
                "metadata": metas_list[i],
                "distance": dists_list[i]
            })

        return contexts

    def format_context_for_prompt(self, contexts: List[Dict[str, Any]]) -> str:
        """Format retrieved contexts into a string suitable for prompt injection."""
        if not contexts:
            return ""

        formatted = "=== RELEVANT CONTEXT ===\n\n"

        for i, ctx in enumerate(contexts, 1):
            meta = ctx["metadata"]
            question = meta.get("question", "")
            source = meta.get("source_file", "unknown")
            formatted += f"[{i}] From {source} - {question}\n"
            formatted += f"{ctx['text']}\n\n"

        return formatted

    def clear_collection(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("✓ Collection cleared")


# Convenience function
def get_context_for_prompt(prompt: str, rag_helper: RAGHelper, n_results: int = 5) -> str:
    """Convenience function to get formatted context for a prompt."""
    contexts = rag_helper.query_context(prompt, n_results=n_results)
    return rag_helper.format_context_for_prompt(contexts)
