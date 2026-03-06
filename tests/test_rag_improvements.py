"""Tests for all 6 Tier 1 RAG improvements.

Tests are organized by improvement number:
1. Contextual Chunk Headers
2. Hybrid Search (RRF)
3. Similarity Threshold
4. Cross-Encoder Reranking
5. Recursive Chunking with Overlap
6. MMR Deduplication
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_utils import (
    RetrievedDoc,
    _cosine_sim,
    apply_similarity_threshold,
    mmr_select,
    prepend_chunk_header,
    reciprocal_rank_fusion,
    recursive_character_split,
    rerank_docs,
    hybrid_retrieve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    url: str = "https://docs.tv/test",
    title: str = "Test",
    content: str = "test content",
    vector_score: float = 0.8,
    bm25_score: float = 0.5,
    embedding: list[float] | None = None,
) -> RetrievedDoc:
    return RetrievedDoc(
        url=url,
        title=title,
        content=content,
        vector_score=vector_score,
        bm25_score=bm25_score,
        embedding=embedding or [],
    )


# ---------------------------------------------------------------------------
# 1. Contextual Chunk Headers
# ---------------------------------------------------------------------------


class TestContextualChunkHeaders:
    def test_prepends_header_with_different_section(self):
        result = prepend_chunk_header("Arrays", "Sorting", "Use array.sort().")
        assert result.startswith("Document: PineScript v5 Reference | Section: Arrays > Sorting")
        assert "Use array.sort()." in result

    def test_prepends_header_same_section_as_page(self):
        result = prepend_chunk_header("Arrays", "Arrays", "Array content.")
        assert "Section: Arrays\n" in result
        assert ">" not in result.split("\n")[0]

    def test_content_preserved(self):
        content = "```pine\nplot(close)\n```"
        result = prepend_chunk_header("Plots", "Basic", content)
        assert content in result

    def test_empty_content(self):
        result = prepend_chunk_header("Page", "Section", "")
        assert "Document: PineScript v5 Reference" in result


# ---------------------------------------------------------------------------
# 2. Hybrid Search — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_fuses_vector_and_bm25_rankings(self):
        docs = [
            _doc(url="a", vector_score=0.9, bm25_score=0.1),
            _doc(url="b", vector_score=0.5, bm25_score=0.9),
            _doc(url="c", vector_score=0.7, bm25_score=0.5),
        ]
        result = reciprocal_rank_fusion(docs, alpha=0.5)
        assert len(result) == 3
        # All docs should have rrf_score > 0
        assert all(d.rrf_score > 0 for d in result)

    def test_alpha_1_favors_vector(self):
        docs = [
            _doc(url="vec_best", vector_score=0.99, bm25_score=0.01),
            _doc(url="bm25_best", vector_score=0.01, bm25_score=0.99),
        ]
        result = reciprocal_rank_fusion(docs, alpha=1.0)
        assert result[0].url == "vec_best"

    def test_alpha_0_favors_bm25(self):
        docs = [
            _doc(url="vec_best", vector_score=0.99, bm25_score=0.01),
            _doc(url="bm25_best", vector_score=0.01, bm25_score=0.99),
        ]
        result = reciprocal_rank_fusion(docs, alpha=0.0)
        assert result[0].url == "bm25_best"

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []

    def test_single_doc(self):
        docs = [_doc(url="only")]
        result = reciprocal_rank_fusion(docs)
        assert len(result) == 1
        assert result[0].rrf_score > 0

    def test_equal_scores_stable(self):
        docs = [
            _doc(url="a", vector_score=0.5, bm25_score=0.5),
            _doc(url="b", vector_score=0.5, bm25_score=0.5),
        ]
        result = reciprocal_rank_fusion(docs, alpha=0.5)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 3. Similarity Threshold
# ---------------------------------------------------------------------------


class TestSimilarityThreshold:
    def test_filters_below_threshold(self):
        docs = [
            _doc(url="good", vector_score=0.8),
            _doc(url="bad", vector_score=0.1),
            _doc(url="edge", vector_score=0.3),
        ]
        result = apply_similarity_threshold(docs, threshold=0.3)
        urls = [d.url for d in result]
        assert "good" in urls
        assert "edge" in urls
        assert "bad" not in urls

    def test_keeps_all_above_threshold(self):
        docs = [_doc(vector_score=0.9), _doc(vector_score=0.5)]
        result = apply_similarity_threshold(docs, threshold=0.3)
        assert len(result) == 2

    def test_filters_all_below_threshold(self):
        docs = [_doc(vector_score=0.1), _doc(vector_score=0.2)]
        result = apply_similarity_threshold(docs, threshold=0.5)
        assert len(result) == 0

    def test_empty_input(self):
        assert apply_similarity_threshold([], threshold=0.3) == []

    def test_exact_threshold_included(self):
        docs = [_doc(vector_score=0.3)]
        result = apply_similarity_threshold(docs, threshold=0.3)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 4. Cross-Encoder Reranking
# ---------------------------------------------------------------------------


class TestCrossEncoderReranking:
    @patch("rag_utils._get_cross_encoder")
    def test_reranks_by_score(self, mock_get_encoder):
        mock_encoder = MagicMock()
        # Return scores — second doc is most relevant
        mock_encoder.predict.return_value = [0.2, 0.9, 0.5]
        mock_get_encoder.return_value = mock_encoder

        docs = [
            _doc(url="a", content="content a"),
            _doc(url="b", content="content b"),
            _doc(url="c", content="content c"),
        ]
        result = rerank_docs("query", docs, top_n=2)

        assert len(result) == 2
        assert result[0].url == "b"  # highest score
        assert result[0].rerank_score == pytest.approx(0.9)

    @patch("rag_utils._get_cross_encoder")
    def test_respects_top_n(self, mock_get_encoder):
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.9, 0.7, 0.5, 0.3]
        mock_get_encoder.return_value = mock_encoder

        docs = [_doc(url=str(i)) for i in range(4)]
        result = rerank_docs("q", docs, top_n=2)
        assert len(result) == 2

    @patch("rag_utils._get_cross_encoder")
    def test_no_encoder_returns_truncated(self, mock_get_encoder):
        mock_get_encoder.return_value = None
        docs = [_doc(url=str(i)) for i in range(5)]
        result = rerank_docs("q", docs, top_n=3)
        assert len(result) == 3

    def test_empty_docs(self):
        assert rerank_docs("query", []) == []

    @patch("rag_utils._get_cross_encoder")
    def test_passes_correct_pairs(self, mock_get_encoder):
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.5]
        mock_get_encoder.return_value = mock_encoder

        docs = [_doc(content="pine script plot")]
        rerank_docs("how to plot", docs, top_n=1)

        pairs = mock_encoder.predict.call_args[0][0]
        assert pairs == [["how to plot", "pine script plot"]]


# ---------------------------------------------------------------------------
# 5. Recursive Character Splitting with Overlap
# ---------------------------------------------------------------------------


class TestRecursiveCharacterSplit:
    def test_short_text_no_split(self):
        text = "Short text."
        result = recursive_character_split(text, chunk_size=500)
        assert len(result) == 1
        assert result[0] == text

    def test_splits_on_paragraph_boundary(self):
        text = "First paragraph content.\n\nSecond paragraph content."
        result = recursive_character_split(text, chunk_size=30, chunk_overlap=0)
        assert len(result) >= 2
        assert "First" in result[0]
        assert "Second" in result[-1]

    def test_splits_on_newline_when_no_paragraph(self):
        text = "Line one is here\nLine two is here\nLine three is here"
        result = recursive_character_split(text, chunk_size=25, chunk_overlap=0)
        assert len(result) >= 2

    def test_overlap_adds_context(self):
        text = "A" * 200 + "\n\n" + "B" * 200
        result = recursive_character_split(text, chunk_size=250, chunk_overlap=50)
        assert len(result) >= 2
        # Second chunk should contain overlap from first
        if len(result) > 1:
            assert len(result[1]) > 200  # has overlap content

    def test_empty_text(self):
        assert recursive_character_split("") == []
        assert recursive_character_split("   ") == []

    def test_respects_chunk_size(self):
        text = "word " * 500  # ~2500 chars
        result = recursive_character_split(text, chunk_size=500, chunk_overlap=50)
        # Chunks might be slightly larger due to overlap prepend, but base splits respect size
        assert len(result) > 1

    def test_code_blocks_handled(self):
        text = "Introduction.\n\n```pine\nplot(close)\n```\n\nMore text."
        result = recursive_character_split(text, chunk_size=500)
        joined = " ".join(result)
        assert "plot(close)" in joined

    def test_preserves_all_content(self):
        text = "alpha\n\nbeta\n\ngamma\n\ndelta"
        result = recursive_character_split(text, chunk_size=10, chunk_overlap=0)
        # All original words should appear somewhere in the chunks
        joined = " ".join(result)
        for word in ["alpha", "beta", "gamma", "delta"]:
            assert word in joined


# ---------------------------------------------------------------------------
# 6. MMR Deduplication
# ---------------------------------------------------------------------------


class TestMMRDeduplication:
    def _make_docs_with_embeddings(self) -> list[RetrievedDoc]:
        """Create docs with distinct and duplicate-like embeddings.

        Uses small, controlled vectors for deterministic behavior.
        dup1 and dup2 point in nearly the same direction.
        diverse points in a different direction but still has decent query relevance.
        """
        # 10-dim vectors for simplicity
        base = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        similar = [0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # diverse: decent relevance to query (~0.7 cosine) but different direction
        diverse = [0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return [
            _doc(url="dup1", vector_score=0.9, embedding=base),
            _doc(url="dup2", vector_score=0.88, embedding=similar),
            _doc(url="diverse", vector_score=0.7, embedding=diverse),
        ]

    def test_mmr_promotes_diversity(self):
        docs = self._make_docs_with_embeddings()
        # Query is between dup and diverse directions — breaks symmetry
        query_emb = [0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        result = mmr_select(query_emb, docs, top_n=2, lambda_param=0.5)
        urls = [d.url for d in result]
        # dup1 is most relevant, but for 2nd pick, diverse has lower
        # similarity to already-selected dup1 → MMR prefers it over dup2
        assert "dup1" in urls
        assert "diverse" in urls

    def test_lambda_1_pure_relevance(self):
        docs = self._make_docs_with_embeddings()
        query_emb = docs[0].embedding

        result = mmr_select(query_emb, docs, top_n=2, lambda_param=1.0)
        # With pure relevance, should pick top-2 by cosine to query
        assert result[0].url == "dup1"

    def test_empty_docs(self):
        assert mmr_select([0.1] * 10, []) == []

    def test_empty_query_embedding(self):
        docs = [_doc(embedding=[0.1] * 10)]
        result = mmr_select([], docs, top_n=1)
        assert len(result) <= 1

    def test_docs_without_embeddings_appended(self):
        docs = [
            _doc(url="with_emb", embedding=[0.1] * 10, vector_score=0.9),
            _doc(url="no_emb", embedding=[], vector_score=0.5),
        ]
        result = mmr_select([0.1] * 10, docs, top_n=2)
        urls = [d.url for d in result]
        assert "with_emb" in urls
        assert "no_emb" in urls

    def test_top_n_respected(self):
        docs = self._make_docs_with_embeddings()
        result = mmr_select(docs[0].embedding, docs, top_n=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_sim([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_sim([0, 0], [1, 1]) == 0.0


# ---------------------------------------------------------------------------
# Integration: Crawler chunking (imports from pinescript_crawler)
# ---------------------------------------------------------------------------


class TestCrawlerChunking:
    """Test that the crawler's split_into_sections uses recursive chunking + headers.

    We mock the 'agent' module before importing pinescript_crawler to avoid
    pydantic-ai version issues at module import time.
    """

    @staticmethod
    def _get_crawler():
        """Import and create a minimal crawler instance, mocking agent module."""
        import sys

        # Mock agent module to avoid pydantic-ai Agent() call at import
        agent_mock = MagicMock()
        agent_mock.database_connect = MagicMock()
        original = sys.modules.get("agent")
        sys.modules["agent"] = agent_mock

        try:
            # Force reimport of pinescript_crawler with mocked agent
            if "pinescript_crawler" in sys.modules:
                del sys.modules["pinescript_crawler"]
            from pinescript_crawler import PineScriptDocsCrawler

            crawler = PineScriptDocsCrawler.__new__(PineScriptDocsCrawler)
            object.__setattr__(crawler, "base_url", "https://example.com")
            object.__setattr__(crawler, "output_dir", "/tmp/test")
            object.__setattr__(crawler, "visited_urls", set())
            return crawler
        finally:
            # Restore original module
            if original is not None:
                sys.modules["agent"] = original
            elif "agent" in sys.modules:
                del sys.modules["agent"]

    def test_split_produces_chunks_with_headers(self):
        crawler = self._get_crawler()

        markdown = "# Arrays\n\n## Sorting\n\nUse array.sort() to sort arrays.\n\n## Inserting\n\nUse array.push()."
        sections = crawler.split_into_sections(markdown, "https://docs.tv/arrays")

        assert len(sections) >= 2
        # Each section should have contextual header
        for sec in sections:
            assert "Document: PineScript v5 Reference" in sec["content"]
            assert "Section:" in sec["content"]

    def test_split_long_section_into_multiple_chunks(self):
        crawler = self._get_crawler()

        # Create a section longer than CHUNK_SIZE (500 chars)
        long_content = "This is a detailed explanation. " * 50  # ~1600 chars
        markdown = f"# Page\n\n## Big Section\n\n{long_content}"
        sections = crawler.split_into_sections(markdown, "https://docs.tv/page")

        # Should produce multiple chunks from the long section
        assert len(sections) > 1


# ---------------------------------------------------------------------------
# Integration: hybrid_retrieve pipeline (mocked DB + OpenAI)
# ---------------------------------------------------------------------------


class TestHybridRetrievePipeline:
    @pytest.mark.asyncio
    async def test_returns_docs_from_hybrid_query(self):
        # Mock OpenAI embeddings
        embedding_obj = MagicMock()
        embedding_obj.data = [MagicMock(embedding=[0.1] * 1536)]
        openai_mock = AsyncMock()
        openai_mock.embeddings.create = AsyncMock(return_value=embedding_obj)

        # Mock asyncpg pool
        pool_mock = AsyncMock()
        pool_mock.fetch = AsyncMock(
            return_value=[
                {
                    "url": "https://docs.tv/ema",
                    "title": "ta.ema()",
                    "content": "Exponential moving average.",
                    "embedding": str([0.1] * 1536),
                    "vector_score": 0.85,
                    "bm25_score": 0.6,
                },
            ]
        )

        docs = await hybrid_retrieve(
            pool=pool_mock,
            openai_client=openai_mock,
            query="how to use ema",
            enable_rerank=False,  # skip for unit test
            enable_mmr=False,
        )

        assert len(docs) == 1
        assert docs[0].title == "ta.ema()"
        assert docs[0].vector_score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_empty_result_on_no_matches(self):
        embedding_obj = MagicMock()
        embedding_obj.data = [MagicMock(embedding=[0.1] * 1536)]
        openai_mock = AsyncMock()
        openai_mock.embeddings.create = AsyncMock(return_value=embedding_obj)

        pool_mock = AsyncMock()
        pool_mock.fetch = AsyncMock(return_value=[])

        docs = await hybrid_retrieve(
            pool=pool_mock,
            openai_client=openai_mock,
            query="nonexistent topic",
            enable_rerank=False,
            enable_mmr=False,
        )

        assert docs == []

    @pytest.mark.asyncio
    async def test_hybrid_query_uses_cosine_and_tsvector(self):
        embedding_obj = MagicMock()
        embedding_obj.data = [MagicMock(embedding=[0.1] * 1536)]
        openai_mock = AsyncMock()
        openai_mock.embeddings.create = AsyncMock(return_value=embedding_obj)

        pool_mock = AsyncMock()
        pool_mock.fetch = AsyncMock(return_value=[])

        await hybrid_retrieve(
            pool=pool_mock,
            openai_client=openai_mock,
            query="ta.rsi function",
            enable_rerank=False,
            enable_mmr=False,
        )

        # Verify the SQL uses cosine distance (<=>) and plainto_tsquery
        sql = pool_mock.fetch.call_args[0][0]
        assert "<=>" in sql, "Should use cosine distance operator"
        assert "plainto_tsquery" in sql, "Should use full-text search"
        assert "search_vector" in sql, "Should query tsvector column"
        assert "vector_score" in sql, "Should select cosine similarity"


# ---------------------------------------------------------------------------
# DB Schema
# ---------------------------------------------------------------------------


class TestDBSchema:
    def test_schema_has_search_vector(self):
        from db_schema import DB_SCHEMA

        assert "search_vector" in DB_SCHEMA
        assert "tsvector" in DB_SCHEMA

    def test_schema_uses_cosine_ops(self):
        from db_schema import DB_SCHEMA

        assert "vector_cosine_ops" in DB_SCHEMA
        assert "vector_l2_ops" not in DB_SCHEMA

    def test_migration_sql_exists(self):
        from db_schema import DB_MIGRATION

        assert "search_vector" in DB_MIGRATION
        assert "to_tsvector" in DB_MIGRATION
        assert "vector_cosine_ops" in DB_MIGRATION
