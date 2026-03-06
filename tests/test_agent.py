"""Tests for the RAG retrieve tool — mocked DB + embeddings, no container needed."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import Dependencies, retrieve, pinescript_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_deps(rows=None, use_openrouter=False):
    """Build a Dependencies object with mocked OpenAI client and asyncpg pool."""
    # Mock OpenAI embeddings response
    embedding_obj = MagicMock()
    embedding_obj.data = [MagicMock(embedding=[0.1] * 1536)]

    openai_mock = AsyncMock()
    openai_mock.embeddings.create = AsyncMock(return_value=embedding_obj)

    # Mock asyncpg pool
    pool_mock = AsyncMock()
    pool_mock.fetch = AsyncMock(return_value=rows or [])

    return Dependencies(
        openai=openai_mock,
        pool=pool_mock,
        openrouter_api_key="sk-or-test" if use_openrouter else None,
        use_openrouter=use_openrouter,
    )


def _make_ctx(deps):
    """Build a minimal RunContext-like object for the retrieve tool."""
    ctx = MagicMock()
    ctx.deps = deps
    ctx.custom_data = {}
    return ctx


# ---------------------------------------------------------------------------
# retrieve() tool tests
# ---------------------------------------------------------------------------

class TestRetrieveTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_docs(self) -> None:
        rows = [
            {"url": "https://docs.tv/plot", "title": "plot()", "content": "Plots a line on the chart."},
            {"url": "https://docs.tv/hline", "title": "hline()", "content": "Draws a horizontal line."},
        ]
        deps = _make_deps(rows=rows)
        ctx = _make_ctx(deps)

        result = await retrieve(ctx, "how to plot a line")

        assert "plot()" in result
        assert "hline()" in result
        assert "Plots a line" in result
        assert ctx.custom_data["snippets_used"] == 2

    @pytest.mark.asyncio
    async def test_returns_message_when_no_docs(self) -> None:
        deps = _make_deps(rows=[])
        ctx = _make_ctx(deps)

        result = await retrieve(ctx, "nonexistent topic")

        assert "No relevant documentation found" in result

    @pytest.mark.asyncio
    async def test_calls_openai_embeddings(self) -> None:
        deps = _make_deps(rows=[])
        ctx = _make_ctx(deps)

        await retrieve(ctx, "test query")

        deps.openai.embeddings.create.assert_called_once()
        call_kwargs = deps.openai.embeddings.create.call_args
        assert call_kwargs.kwargs["input"] == "test query"
        assert "embedding" in call_kwargs.kwargs["model"]

    @pytest.mark.asyncio
    async def test_queries_db_with_embedding(self) -> None:
        deps = _make_deps(rows=[])
        ctx = _make_ctx(deps)

        await retrieve(ctx, "indicator question")

        deps.pool.fetch.assert_called_once()
        sql_arg = deps.pool.fetch.call_args[0][0]
        assert "pinescript_docs" in sql_arg
        assert "LIMIT" in sql_arg

    @pytest.mark.asyncio
    async def test_handles_embedding_error(self) -> None:
        deps = _make_deps()
        deps.openai.embeddings.create = AsyncMock(side_effect=Exception("API down"))
        ctx = _make_ctx(deps)

        result = await retrieve(ctx, "test")

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_handles_db_error(self) -> None:
        deps = _make_deps()
        ctx = _make_ctx(deps)
        deps.pool.fetch = AsyncMock(side_effect=Exception("connection refused"))

        result = await retrieve(ctx, "test")

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_single_doc_formatting(self) -> None:
        rows = [
            {"url": "https://docs.tv/var", "title": "Variables", "content": "Use var to declare."},
        ]
        deps = _make_deps(rows=rows)
        ctx = _make_ctx(deps)

        result = await retrieve(ctx, "variables")

        assert "# Variables" in result
        assert "https://docs.tv/var" in result
        assert ctx.custom_data["snippets_used"] == 1


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------

class TestAgentWiring:
    def test_agent_has_retrieve_tool(self) -> None:
        tool_names = [t.name for t in pinescript_agent._function_tools.values()]
        assert "retrieve" in tool_names

    def test_agent_result_type(self) -> None:
        from agent import PineScriptResult
        assert pinescript_agent.result_type == PineScriptResult

    def test_agent_uses_config_model(self) -> None:
        from config import DEFAULT_MODEL
        # Agent was initialised with DEFAULT_MODEL from config
        assert DEFAULT_MODEL == "openai:gpt-4o-mini"


# ---------------------------------------------------------------------------
# Dependencies — OpenRouter routing
# ---------------------------------------------------------------------------

class TestOpenRouterRouting:
    def test_deps_without_openrouter(self) -> None:
        deps = _make_deps(use_openrouter=False)
        assert deps.use_openrouter is False
        assert deps.openrouter_api_key is None

    def test_deps_with_openrouter(self) -> None:
        deps = _make_deps(use_openrouter=True)
        assert deps.use_openrouter is True
        assert deps.openrouter_api_key == "sk-or-test"
