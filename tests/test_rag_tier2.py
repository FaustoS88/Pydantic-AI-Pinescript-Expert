"""Tests for Tier 2 RAG improvements.

Covers:
- code_aware_split: 8 tests
- generate_contextual_prefix: 4 tests (mocked LLM)
- detect_content_type: 4 tests
- DB schema Tier 2 columns: 4 tests
"""

from __future__ import annotations

import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_utils import code_aware_split, detect_content_type, generate_contextual_prefix
from db_schema import DB_SCHEMA, DB_MIGRATION


# ===========================================================================
# TestCodeAwareSplit
# ===========================================================================


class TestCodeAwareSplit:
    """code_aware_split never splits inside fenced code blocks."""

    def test_short_text_no_split(self):
        """Text shorter than chunk_size returns a single chunk."""
        text = "This is a short text."
        result = code_aware_split(text, chunk_size=1000)
        assert result == ["This is a short text."]

    def test_empty_text(self):
        """Empty string returns empty list."""
        assert code_aware_split("") == []
        assert code_aware_split("   ") == []

    def test_prose_splits_normally(self):
        """Long prose text is split into multiple chunks."""
        # 10 paragraphs of 120 chars each = 1200 chars total
        paragraph = "A" * 100 + "\n\n"
        text = paragraph * 10
        result = code_aware_split(text, chunk_size=250, chunk_overlap=0)
        assert len(result) > 1
        # All chunks should be prose (no triple backticks)
        for chunk in result:
            assert "```" not in chunk

    def test_code_block_never_split(self):
        """A single code block that fits within chunk_size is kept whole."""
        code = "```pine\nindicator('My Script', overlay=true)\nplot(close)\n```"
        text = "Some intro.\n\n" + code + "\n\nSome outro."
        result = code_aware_split(text, chunk_size=500)
        # Code block must appear intact in exactly one chunk
        code_chunks = [c for c in result if "```pine" in c]
        assert len(code_chunks) == 1
        assert "indicator('My Script', overlay=true)" in code_chunks[0]
        assert "plot(close)" in code_chunks[0]

    def test_code_block_larger_than_chunk_size_kept_whole(self):
        """A code block larger than chunk_size is returned as a single oversized chunk."""
        long_code = "```pine\n" + "plot(close) // line\n" * 100 + "```"
        assert len(long_code) > 500
        result = code_aware_split(long_code, chunk_size=200, chunk_overlap=0)
        # Code block must appear intact (not split into pieces)
        code_chunks = [c for c in result if c.startswith("```pine")]
        assert len(code_chunks) == 1
        # The single chunk must contain all original code lines
        assert long_code.strip() in result

    def test_mixed_prose_and_code(self):
        """Prose and code blocks are separated correctly."""
        text = (
            "First paragraph about functions.\n\n"
            "```pine\nf(x) => x * 2\n```\n\n"
            "Second paragraph explaining the result.\n\n"
            "```pine\nplot(f(close))\n```"
        )
        result = code_aware_split(text, chunk_size=100, chunk_overlap=0)
        # No chunk should have a code block split across it
        for chunk in result:
            backtick_count = chunk.count("```")
            # Even count means balanced fences (or zero)
            assert backtick_count % 2 == 0, (
                f"Unbalanced backticks in chunk: {chunk!r}"
            )

    def test_inline_code_not_protected(self):
        """Single-backtick inline code is NOT treated as a fenced block."""
        long_text = (
            "The `ta.ema()` function computes the exponential moving average. " * 30
        )
        result = code_aware_split(long_text, chunk_size=200, chunk_overlap=0)
        # Should split into multiple chunks (inline code does not block splitting)
        assert len(result) > 1

    def test_multiple_code_blocks(self):
        """Multiple code blocks — none are split."""
        blocks = [f"```pine\nblock_{i} = {i}\n```" for i in range(5)]
        text = "\n\nSome prose between blocks.\n\n".join(blocks)
        result = code_aware_split(text, chunk_size=80, chunk_overlap=0)
        for chunk in result:
            backtick_count = chunk.count("```")
            assert backtick_count % 2 == 0, (
                f"Unbalanced backticks in chunk: {chunk!r}"
            )
        # Each code block must appear in its entirety somewhere
        for i in range(5):
            assert any(f"block_{i} = {i}" in c for c in result)


# ===========================================================================
# TestGenerateContextualPrefix
# ===========================================================================


class TestGenerateContextualPrefix:
    """generate_contextual_prefix: LLM-generated 1-sentence chunk description."""

    def _make_mock_client(self, return_text: str):
        """Build an AsyncOpenAI-like mock that returns return_text."""
        mock_choice = MagicMock()
        mock_choice.message.content = return_text

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        mock_completions = AsyncMock(return_value=mock_resp)

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = mock_completions

        return mock_client

    @pytest.mark.asyncio
    async def test_returns_prefix_string(self):
        """Returns the LLM-generated sentence as a string."""
        expected = "PineScript ta.ema() calculates the Exponential Moving Average."
        client = self._make_mock_client(expected)

        result = await generate_contextual_prefix(
            chunk="ema = ta.ema(close, 14)",
            page_title="Technical Analysis",
            full_page_content="Full page about ta functions...",
            openai_client=client,
            model="openai/gpt-4.1-mini",
        )
        assert result == expected
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_truncates_large_page_content(self):
        """Prompt is constructed even with very large page content (no crash)."""
        large_page = "x" * 100_000  # 100k chars — much larger than MAX_PAGE_CONTEXT_CHARS
        client = self._make_mock_client("PineScript describes something.")

        # Should not raise even with oversized page content
        result = await generate_contextual_prefix(
            chunk="some chunk",
            page_title="Big Page",
            full_page_content=large_page,
            openai_client=client,
        )
        assert isinstance(result, str)

        # Verify truncation: the prompt sent to LLM must NOT contain all 100k chars.
        # We check the LLM was called exactly once (no crash).
        client.chat.completions.create.assert_awaited_once()
        call_kwargs = client.chat.completions.create.call_args
        prompt_text = call_kwargs.kwargs.get("messages", call_kwargs.args[0] if call_kwargs.args else [])
        # Extract prompt content from messages list
        if isinstance(prompt_text, list):
            full_prompt = " ".join(m.get("content", "") for m in prompt_text)
            assert len(full_prompt) < len(large_page)

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_page_title(self):
        """On LLM failure, returns page_title as fallback (no crash)."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

        result = await generate_contextual_prefix(
            chunk="some chunk",
            page_title="Technical Analysis Functions",
            full_page_content="page content",
            openai_client=mock_client,
        )
        assert result == "Technical Analysis Functions"

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_llm_response(self):
        """Leading/trailing whitespace in LLM response is stripped."""
        client = self._make_mock_client("  PineScript describes ta.rsi().  ")

        result = await generate_contextual_prefix(
            chunk="chunk text",
            page_title="RSI",
            full_page_content="page",
            openai_client=client,
        )
        assert result == "PineScript describes ta.rsi()."
        assert not result.startswith(" ")
        assert not result.endswith(" ")


# ===========================================================================
# TestDetectContentType
# ===========================================================================


class TestDetectContentType:
    """detect_content_type returns 'reference' | 'example' | 'tutorial'."""

    def test_code_heavy_returns_example(self):
        """2+ code blocks AND short text → 'example'."""
        content = "```pine\na = 1\n```\n\nSome note.\n\n```pine\nb = 2\n```"
        # word count should be < 200
        assert len(content.split()) < 200
        assert detect_content_type(content) == "example"

    def test_function_reference_returns_reference(self):
        """Has param keywords AND exactly one code block → 'reference'.

        We use exactly 1 code block so the `example` condition (>= 2 blocks AND
        < 200 words) does not trigger, allowing `reference` to be returned.
        """
        content = (
            "## ta.ema(source, length)\n\n"
            "The ta.ema() function computes the Exponential Moving Average. "
            "**Parameter** source: Series of values to process. Usually close. "
            "**Parameter** length: Number of bars (look-back period). "
            "**Returns**: A float series representing the EMA values. "
            "**Syntax**: ta.ema(source, length) where source is a series and "
            "length is a simple integer. The return type is float. "
            "Note that this function requires at least `length` bars to produce "
            "a valid result; earlier bars will return na.\n\n"
            "```pine\nema14 = ta.ema(close, 14)\nplot(ema14)\n```"
        )
        # Exactly 1 code block — 'example' requires >= 2
        assert content.count("```") // 2 == 1
        assert detect_content_type(content) == "reference"

    def test_prose_returns_tutorial(self):
        """No code blocks and no param keywords → 'tutorial'."""
        content = (
            "PineScript is a scripting language designed for traders. "
            "It allows you to create custom indicators and strategies. "
            "You can plot values on the chart and draw shapes. " * 20
        )
        assert "```" not in content
        assert detect_content_type(content) == "tutorial"

    def test_empty_content(self):
        """Empty string returns 'tutorial' (safe default)."""
        assert detect_content_type("") == "tutorial"
        assert detect_content_type("   ") == "tutorial"


# ===========================================================================
# TestDBSchemaT2
# ===========================================================================


class TestDBSchemaT2:
    """Verify Tier 2 metadata columns appear in DB_SCHEMA and DB_MIGRATION."""

    def test_schema_has_chunk_index(self):
        assert "chunk_index" in DB_SCHEMA

    def test_schema_has_content_type(self):
        assert "content_type" in DB_SCHEMA

    def test_schema_has_contextual_prefix(self):
        assert "contextual_prefix" in DB_SCHEMA

    def test_migration_has_all_new_columns(self):
        assert "chunk_index" in DB_MIGRATION
        assert "content_type" in DB_MIGRATION
        assert "contextual_prefix" in DB_MIGRATION
