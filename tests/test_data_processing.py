"""
Test: Data Processing
=====================
Verify the system can load and parse the new flat Q&A JSON format.
"""

import sys
import json
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_helper import RAGHelper


class TestLoadQAJson:
    """Tests for RAGHelper.load_qa_json()."""

    def test_loads_valid_file(self, sample_qa_file, sample_qa_data):
        result = RAGHelper.load_qa_json(sample_qa_file)
        assert len(result) == len(sample_qa_data)

    def test_returns_correct_structure(self, sample_qa_file):
        result = RAGHelper.load_qa_json(sample_qa_file)
        for item in result:
            assert "question" in item
            assert "answer" in item
            assert isinstance(item["question"], str)
            assert isinstance(item["answer"], str)

    def test_preserves_content(self, sample_qa_file):
        result = RAGHelper.load_qa_json(sample_qa_file)
        questions = [item["question"] for item in result]
        assert "What is your full name?" in questions

    def test_skips_empty_answers(self, qa_with_empty_answers):
        result = RAGHelper.load_qa_json(qa_with_empty_answers)
        # Only "Isaac" and "Software Engineer" have non-empty answers
        assert len(result) == 2
        answers = [item["answer"] for item in result]
        assert "Isaac" in answers
        assert "Software Engineer" in answers

    def test_empty_list_returns_empty(self, empty_qa_file):
        result = RAGHelper.load_qa_json(empty_qa_file)
        assert result == []

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            RAGHelper.load_qa_json("/nonexistent/path/fake.json")

    def test_invalid_json_structure_raises(self, tmp_path):
        """A JSON object (dict) instead of array should raise ValueError."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"not": "an array"}')
        with pytest.raises(ValueError, match="Expected a JSON array"):
            RAGHelper.load_qa_json(str(bad_file))

    def test_missing_keys_raises(self, tmp_path):
        """Items without 'question' or 'answer' keys should raise ValueError."""
        bad_file = tmp_path / "bad_keys.json"
        bad_file.write_text('[{"topic": "test", "info": "data"}]')
        with pytest.raises(ValueError, match="missing 'question' or 'answer'"):
            RAGHelper.load_qa_json(str(bad_file))

    def test_non_dict_items_raise(self, tmp_path):
        """Items that are strings instead of dicts should raise ValueError."""
        bad_file = tmp_path / "bad_items.json"
        bad_file.write_text('["just a string"]')
        with pytest.raises(ValueError, match="not a dict"):
            RAGHelper.load_qa_json(str(bad_file))

    def test_strips_whitespace(self, tmp_path):
        """Questions and answers should be stripped of leading/trailing whitespace."""
        data = [{"question": "  Name?  ", "answer": "  Isaac  "}]
        file_path = tmp_path / "whitespace.json"
        file_path.write_text(json.dumps(data))
        result = RAGHelper.load_qa_json(str(file_path))
        assert result[0]["question"] == "Name?"
        assert result[0]["answer"] == "Isaac"
