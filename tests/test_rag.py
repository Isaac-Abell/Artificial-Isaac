"""
Test: RAG Pipeline
==================
Verify the embedding and retrieval functions return correct context for a query.
Uses a temporary ChromaDB instance so it doesn't pollute your real database.
"""

import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_helper import RAGHelper, get_context_for_prompt


@pytest.fixture
def rag(tmp_path, sample_qa_file):
    """Create a RAGHelper with a temp ChromaDB and index sample data."""
    helper = RAGHelper(
        persist_directory=str(tmp_path / "chroma_test"),
        collection_name="test_collection"
    )
    helper.index_qa_file(sample_qa_file)
    return helper


class TestIndexing:
    """Tests for indexing Q&A data into ChromaDB."""

    def test_index_creates_documents(self, rag, sample_qa_data):
        count = rag.collection.count()
        assert count == len(sample_qa_data)

    def test_index_empty_file_returns_zero(self, tmp_path, empty_qa_file):
        helper = RAGHelper(
            persist_directory=str(tmp_path / "chroma_empty"),
            collection_name="empty_test"
        )
        count = helper.index_qa_file(empty_qa_file)
        assert count == 0
        assert helper.collection.count() == 0

    def test_index_skips_empty_answers(self, tmp_path, qa_with_empty_answers):
        helper = RAGHelper(
            persist_directory=str(tmp_path / "chroma_partial"),
            collection_name="partial_test"
        )
        count = helper.index_qa_file(qa_with_empty_answers)
        assert count == 2  # Only 2 non-empty answers


class TestQuery:
    """Tests for querying the RAG system."""

    def test_query_returns_results(self, rag):
        results = rag.query_context("What is your name?", n_results=3)
        assert len(results) > 0

    def test_query_returns_relevant_result(self, rag):
        results = rag.query_context("What is your name?", n_results=1)
        # The top result should be about the name
        top_text = results[0]["text"]
        assert "Isaac" in top_text

    def test_query_result_structure(self, rag):
        results = rag.query_context("mountain biking", n_results=1)
        result = results[0]
        assert "id" in result
        assert "text" in result
        assert "metadata" in result
        assert "distance" in result

    def test_query_metadata_has_expected_fields(self, rag):
        results = rag.query_context("programming", n_results=1)
        meta = results[0]["metadata"]
        assert "source_file" in meta
        assert "question" in meta
        assert "chunk_type" in meta
        assert meta["chunk_type"] == "qa_pair"

    def test_query_empty_collection(self, tmp_path):
        helper = RAGHelper(
            persist_directory=str(tmp_path / "chroma_empty_q"),
            collection_name="empty_query_test"
        )
        results = helper.query_context("anything", n_results=3)
        assert results == []


class TestFormatContext:
    """Tests for formatting context into prompt strings."""

    def test_format_empty_returns_empty_string(self, rag):
        result = rag.format_context_for_prompt([])
        assert result == ""

    def test_format_includes_header(self, rag):
        contexts = rag.query_context("name", n_results=1)
        formatted = rag.format_context_for_prompt(contexts)
        assert "RELEVANT CONTEXT" in formatted

    def test_format_includes_text(self, rag):
        contexts = rag.query_context("name", n_results=1)
        formatted = rag.format_context_for_prompt(contexts)
        assert "Isaac" in formatted

    def test_convenience_function(self, rag):
        result = get_context_for_prompt("name", rag, n_results=1)
        assert isinstance(result, str)
        assert "Isaac" in result


class TestClearCollection:
    """Tests for clearing the collection."""

    def test_clear_removes_all_documents(self, rag):
        assert rag.collection.count() > 0
        rag.clear_collection()
        assert rag.collection.count() == 0
