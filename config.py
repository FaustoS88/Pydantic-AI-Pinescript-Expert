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
