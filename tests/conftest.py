"""
Shared pytest fixtures for Artificial Isaac tests.
"""

import pytest
import json
import tempfile
from pathlib import Path


@pytest.fixture
def sample_qa_data():
    """Returns a list of sample Q&A pairs."""
    return [
        {"question": "What is your full name?", "answer": "Isaac Abell"},
        {"question": "Where did you grow up?", "answer": "I grew up in Seattle, WA. I loved the rain but hated the traffic."},
        {"question": "What is your favorite sport?", "answer": "Mountain biking. I ride a Propain Spindrift."},
        {"question": "What programming languages do you know?", "answer": "Python, JavaScript, and some Rust."},
        {"question": "Are you a morning person or night owl?", "answer": "Definitely a night owl."},
    ]


@pytest.fixture
def sample_qa_file(sample_qa_data, tmp_path):
    """Creates a temporary Q&A JSON file and returns its path."""
    file_path = tmp_path / "biography.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_qa_data, f, indent=2)
    return str(file_path)


@pytest.fixture
def empty_qa_file(tmp_path):
    """Creates a temporary empty Q&A JSON file (empty list)."""
    file_path = tmp_path / "empty.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([], f)
    return str(file_path)


@pytest.fixture
def qa_with_empty_answers(tmp_path):
    """Creates a Q&A JSON file where some answers are empty."""
    data = [
        {"question": "What is your name?", "answer": "Isaac"},
        {"question": "What is your favorite color?", "answer": ""},
        {"question": "Where do you live?", "answer": "   "},
        {"question": "What is your job?", "answer": "Software Engineer"},
    ]
    file_path = tmp_path / "partial.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(file_path)
