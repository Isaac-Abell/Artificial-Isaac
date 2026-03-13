"""
Test: Tokenization
==================
Verify the model's tokenizer correctly processes standard and edge-case inputs.

Note: This test downloads the tokenizer on first run (~500MB for Qwen).
      Set SKIP_TOKENIZER_TESTS=1 to skip if you don't have it cached.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Skip all tests in this file if env var is set (for CI or no-GPU setups)
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_TOKENIZER_TESTS") == "1",
    reason="Skipping tokenizer tests (SKIP_TOKENIZER_TESTS=1)"
)


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tokenizer once for all tests in this module."""
    from transformers import AutoTokenizer
    from config import TOKENIZER_ID

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True, use_fast=True)
    return tok


class TestTokenizerLoads:
    """Basic tokenizer loading and properties."""

    def test_tokenizer_is_not_none(self, tokenizer):
        assert tokenizer is not None

    def test_tokenizer_has_vocab(self, tokenizer):
        assert tokenizer.vocab_size > 0

    def test_pad_token_can_be_set(self, tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token is not None


class TestTokenizerEncoding:
    """Encoding and decoding tests."""

    def test_encode_simple_text(self, tokenizer):
        text = "Hello, my name is Isaac."
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_decode_roundtrip(self, tokenizer):
        text = "I grew up in Seattle and love mountain biking."
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        # Decoded text should be semantically equivalent (whitespace may differ)
        assert "Seattle" in decoded
        assert "mountain biking" in decoded

    def test_encode_empty_string(self, tokenizer):
        tokens = tokenizer.encode("", add_special_tokens=False)
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_encode_special_characters(self, tokenizer):
        text = "Hello! 🚀 こんにちは @user #hashtag $100 <tag>"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert len(tokens) > 0

    def test_encode_long_text(self, tokenizer):
        text = "This is a test sentence. " * 500  # ~2500 tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert len(tokens) > 100

    def test_batch_encoding(self, tokenizer):
        texts = ["Hello", "World", "How are you?"]
        result = tokenizer(texts, padding=True, return_tensors="pt")
        assert result["input_ids"].shape[0] == 3
