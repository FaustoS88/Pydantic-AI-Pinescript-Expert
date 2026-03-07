"""Centralised configuration for the PineScript Expert agent.

All magic numbers and hardcoded values live here. Override via
environment variables or by editing the presets below.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("PINESCRIPT_MODEL", "openai:gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("PINESCRIPT_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("PINESCRIPT_MAX_TOKENS", "2000"))

# ---------------------------------------------------------------------------
# Embedding settings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv("PINESCRIPT_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = 1536  # fixed for text-embedding-3-small
VECTOR_SEARCH_LIMIT = int(os.getenv("PINESCRIPT_VECTOR_SEARCH_LIMIT", "8"))

# ---------------------------------------------------------------------------
# Advanced RAG settings
# ---------------------------------------------------------------------------

# Hybrid search: weight between vector (1.0) and BM25 (0.0)
HYBRID_SEARCH_ALPHA = float(os.getenv("PINESCRIPT_HYBRID_ALPHA", "0.5"))

# Similarity threshold (cosine): discard results below this score
SIMILARITY_THRESHOLD = float(os.getenv("PINESCRIPT_SIMILARITY_THRESHOLD", "0.3"))

# Retrieval candidates before reranking
RETRIEVAL_CANDIDATES = int(os.getenv("PINESCRIPT_RETRIEVAL_CANDIDATES", "15"))

# Final results after reranking
RERANK_TOP_N = int(os.getenv("PINESCRIPT_RERANK_TOP_N", "15"))

# Cross-encoder model for reranking
RERANK_MODEL = os.getenv(
    "PINESCRIPT_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# MMR diversity parameter (1.0 = pure relevance, 0.0 = pure diversity)
MMR_LAMBDA = float(os.getenv("PINESCRIPT_MMR_LAMBDA", "0.7"))

# Chunking parameters
CHUNK_SIZE = int(os.getenv("PINESCRIPT_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("PINESCRIPT_CHUNK_OVERLAP", "100"))

# ---------------------------------------------------------------------------
# OpenRouter settings
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_DEFAULT_MODEL = os.getenv(
    "OPENROUTER_MODEL", "openai/gpt-4.1-mini"
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DEFAULT_DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres"
)

# ---------------------------------------------------------------------------
# Model presets — quick switch for testing with different providers
# ---------------------------------------------------------------------------

MODEL_PRESETS: dict[str, dict[str, str | float | int]] = {
    "default": {
        "model": "openai/gpt-4.1-mini",
        "temperature": 0.2,
        "max_tokens": 2000,
    },
    "codex": {
        "model": "openai/gpt-5.3-codex",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "opus": {
        "model": "anthropic/claude-opus-4-6",
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "flash": {
        "model": "google/gemini-3-flash-preview",
        "temperature": 0.3,
        "max_tokens": 2000,
    },
}


def get_preset(name: str) -> dict[str, str | float | int]:
    """Return a model preset by name, falling back to 'default'."""
    return MODEL_PRESETS.get(name, MODEL_PRESETS["default"])
