"""Tests for the RAG retrieve tool — mocked retrieval pipeline, no container needed."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import (
    Dependencies,
    PineScriptResult,
    pinescript_agent,
    pinescript_rag,
    resolve_model,
    retrieve,
)
from config import EMBEDDING_MODEL, OPENROUTER_DEFAULT_MODEL
from rag_utils import RetrievedDoc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_deps():
    """Build a Dependencies object with mocked OpenAI client and asyncpg pool."""
    return Dependencies(openai=AsyncMock(), pool=AsyncMock())


def _make_ctx(deps):
    """Build a minimal RunContext-like object for the retrieve tool."""
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


def _doc(url: str, title: str, content: str) -> RetrievedDoc:
    return RetrievedDoc(url=url, title=title, content=content)


# ---------------------------------------------------------------------------
# retrieve() tool tests
#
# retrieve() delegates the whole vector + BM25 → RRF → threshold → rerank →
# MMR pipeline to rag_utils.hybrid_retrieve (covered in test_rag_improvements),
# so these tests patch it and assert the tool's own contract: formatting,
# the empty case, and error handling.
# ---------------------------------------------------------------------------


class TestRetrieveTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_docs(self) -> None:
        docs = [
            _doc("https://docs.tv/plot", "plot()", "Plots a line on the chart."),
            _doc("https://docs.tv/hline", "hline()", "Draws a horizontal line."),
        ]
        ctx = _make_ctx(_make_deps())

        with patch("agent.hybrid_retrieve", AsyncMock(return_value=docs)):
            result = await retrieve(ctx, "how to plot a line")

        assert "plot()" in result
        assert "hline()" in result
        assert "Plots a line" in result
        assert "https://docs.tv/hline" in result

    @pytest.mark.asyncio
    async def test_returns_message_when_no_docs(self) -> None:
        ctx = _make_ctx(_make_deps())

        with patch("agent.hybrid_retrieve", AsyncMock(return_value=[])):
            result = await retrieve(ctx, "nonexistent topic")

        assert "No relevant documentation found" in result

    @pytest.mark.asyncio
    async def test_passes_deps_and_query_to_pipeline(self) -> None:
        deps = _make_deps()
        ctx = _make_ctx(deps)
        fake_retrieve = AsyncMock(return_value=[])

        with patch("agent.hybrid_retrieve", fake_retrieve):
            await retrieve(ctx, "indicator question")

        fake_retrieve.assert_called_once()
        kwargs = fake_retrieve.call_args.kwargs
        assert kwargs["query"] == "indicator question"
        assert kwargs["pool"] is deps.pool
        assert kwargs["openai_client"] is deps.openai
        assert kwargs["embedding_model"] == EMBEDDING_MODEL

    @pytest.mark.asyncio
    async def test_handles_retrieval_error(self) -> None:
        ctx = _make_ctx(_make_deps())

        with patch("agent.hybrid_retrieve", AsyncMock(side_effect=Exception("connection refused"))):
            result = await retrieve(ctx, "test")

        assert "Error" in result
        assert "connection refused" in result

    @pytest.mark.asyncio
    async def test_single_doc_formatting(self) -> None:
        docs = [_doc("https://docs.tv/var", "Variables", "Use var to declare.")]
        ctx = _make_ctx(_make_deps())

        with patch("agent.hybrid_retrieve", AsyncMock(return_value=docs)):
            result = await retrieve(ctx, "variables")

        assert "# Variables" in result
        assert "https://docs.tv/var" in result
        assert "Use var to declare." in result


# ---------------------------------------------------------------------------
# Agent wiring — v2: the tool is registered on the Capability, and the output
# schema is `output_type` (was `result_type` in v1).
# ---------------------------------------------------------------------------


class TestAgentWiring:
    def test_capability_registers_retrieve_tool(self) -> None:
        assert "retrieve" in pinescript_rag.get_toolset().tools

    def test_agent_output_type(self) -> None:
        assert pinescript_agent.output_type == PineScriptResult

    def test_agent_uses_config_model(self) -> None:
        from config import DEFAULT_MODEL

        # Agent was initialised with DEFAULT_MODEL from config
        assert DEFAULT_MODEL == "openai:gpt-4o-mini"


# ---------------------------------------------------------------------------
# Model routing — v2 moved this off Dependencies and onto resolve_model(),
# which reads OPENROUTER_API_KEY at call time.
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_returns_none_without_openrouter_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert resolve_model() is None

    def test_uses_openrouter_default_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        model = resolve_model()
        assert model is not None
        assert model.model_name == OPENROUTER_DEFAULT_MODEL

    def test_known_preset_resolves_to_preset_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from config import get_preset

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        model = resolve_model("flash")
        assert model is not None
        assert model.model_name == str(get_preset("flash")["model"])

    def test_raw_model_id_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        model = resolve_model("vendor/some-raw-model")
        assert model is not None
        assert model.model_name == "vendor/some-raw-model"

    def test_preset_without_key_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert resolve_model("flash") is None
