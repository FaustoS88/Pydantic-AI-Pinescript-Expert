"""Tests for config module — presets, defaults, and env overrides."""

from __future__ import annotations


class TestConfigDefaults:
    def test_default_model(self) -> None:
        from config import DEFAULT_MODEL
        assert DEFAULT_MODEL == "openai:gpt-4o-mini"

    def test_temperature_is_float(self) -> None:
        from config import LLM_TEMPERATURE
        assert isinstance(LLM_TEMPERATURE, float)
        assert 0.0 <= LLM_TEMPERATURE <= 2.0

    def test_max_tokens_is_int(self) -> None:
        from config import LLM_MAX_TOKENS
        assert isinstance(LLM_MAX_TOKENS, int)
        assert LLM_MAX_TOKENS > 0

    def test_embedding_model(self) -> None:
        from config import EMBEDDING_MODEL
        assert EMBEDDING_MODEL == "text-embedding-3-small"

    def test_embedding_dimension(self) -> None:
        from config import EMBEDDING_DIMENSION
        assert EMBEDDING_DIMENSION == 1536

    def test_vector_search_limit(self) -> None:
        from config import VECTOR_SEARCH_LIMIT
        assert isinstance(VECTOR_SEARCH_LIMIT, int)
        assert VECTOR_SEARCH_LIMIT > 0

    def test_openrouter_base_url(self) -> None:
        from config import OPENROUTER_BASE_URL
        assert "openrouter.ai" in OPENROUTER_BASE_URL

    def test_database_url_has_scheme(self) -> None:
        from config import DEFAULT_DATABASE_URL
        assert DEFAULT_DATABASE_URL.startswith("postgresql://")


class TestModelPresets:
    def test_default_preset_exists(self) -> None:
        from config import MODEL_PRESETS
        assert "default" in MODEL_PRESETS

    def test_codex_preset(self) -> None:
        from config import get_preset
        preset = get_preset("codex")
        assert preset["model"] == "openai/gpt-5.3-codex"
        assert preset["temperature"] == 0.1
        assert preset["max_tokens"] == 4096

    def test_opus_preset(self) -> None:
        from config import get_preset
        preset = get_preset("opus")
        assert preset["model"] == "anthropic/claude-opus-4-6"
        assert preset["max_tokens"] == 4096

    def test_flash_preset(self) -> None:
        from config import get_preset
        preset = get_preset("flash")
        assert "gemini" in preset["model"]

    def test_unknown_preset_returns_default(self) -> None:
        from config import get_preset, MODEL_PRESETS
        result = get_preset("nonexistent")
        assert result == MODEL_PRESETS["default"]

    def test_all_presets_have_required_keys(self) -> None:
        from config import MODEL_PRESETS
        for name, preset in MODEL_PRESETS.items():
            assert "model" in preset, f"preset '{name}' missing 'model'"
            assert "temperature" in preset, f"preset '{name}' missing 'temperature'"
            assert "max_tokens" in preset, f"preset '{name}' missing 'max_tokens'"
