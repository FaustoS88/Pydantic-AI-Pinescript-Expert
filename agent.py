"""
Pine Script Expert Agent — Pydantic AI v2.

Architecture:
  - pinescript_rag  : Capability bundling the retrieve tool + expert instructions
  - Thinking        : Built-in capability for extended reasoning
  - Agent           : Thin shell composing the capabilities above

The RAG pipeline (rag_utils.py), database schema (db_schema.py), and
configuration (config.py) are unchanged from v1.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability, Thinking
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import (
    DEFAULT_DATABASE_URL,
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    HYBRID_SEARCH_ALPHA,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MMR_LAMBDA,
    MODEL_PRESETS,
    OPENROUTER_BASE_URL,
    OPENROUTER_DEFAULT_MODEL,
    RERANK_TOP_N,
    RETRIEVAL_CANDIDATES,
    SIMILARITY_THRESHOLD,
    THINKING_EFFORT,
    get_preset,
)
from rag_utils import hybrid_retrieve

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------


def configured_key(name: str) -> str | None:
    """The key in env var `name`, or None when it is missing or a template placeholder."""
    value = os.getenv(name)
    if not value or "your_" in value.lower() or value == "sk-...":
        return None
    return value


def get_openai_api_key() -> str:
    """Get OpenAI API key with validation and interactive prompt if needed."""
    openai_api_key = configured_key("OPENAI_API_KEY")

    if not openai_api_key:
        logger.warning("OPENAI_API_KEY is not set or has a placeholder value.")
        print("Please enter your OpenAI API key:")
        openai_api_key = input("> ")
        if not openai_api_key:
            raise ValueError("OpenAI API key is required to continue.")
        os.environ["OPENAI_API_KEY"] = openai_api_key

        env_path = os.path.join(os.path.dirname(__file__), ".env")
        try:
            from dotenv import set_key

            set_key(env_path, "OPENAI_API_KEY", openai_api_key)
            logger.info("Updated OPENAI_API_KEY in %s", env_path)
        except ImportError:
            logger.warning("Could not update .env — dotenv.set_key not available")

    return openai_api_key


# ---------------------------------------------------------------------------
# Output schema (unchanged)
# ---------------------------------------------------------------------------


class PineScriptResult(BaseModel):
    query: str = Field(description="The original query")
    response: str = Field(description="The generated response")
    snippets_used: int = Field(description="Number of documentation snippets used")


# ---------------------------------------------------------------------------
# Dependencies — simplified: removed dead openrouter_api_key / use_openrouter
# fields that were never read by any tool.
# ---------------------------------------------------------------------------


@dataclass
class Dependencies:
    """Runtime dependencies injected per run."""

    openai: AsyncOpenAI
    pool: asyncpg.Pool


# ---------------------------------------------------------------------------
# Capability: Pine Script RAG
#
# v1 had: monolithic system_prompt + @agent.tool + @agent.system_prompt
# v2 has: one Capability bundling instructions + the retrieve tool.
#
# defer_loading=False because this is the ONLY capability right now.
# When you add more (e.g. syntax-validation, error-explanation), defer
# THOSE and keep this one always-on.
# ---------------------------------------------------------------------------

pinescript_rag = Capability(
    id="pinescript-rag",
    description="Search the official Pine Script v6 documentation and ground answers in it.",
    instructions=(
        "You are a Pine Script v6 expert assistant. Pine Script is the programming "
        "language used in TradingView for creating custom indicators and strategies "
        "for technical analysis of financial markets.\n\n"
        "ALWAYS call the retrieve tool before answering any Pine Script question. "
        "Ground every answer in the official documentation you retrieve. "
        "When generating code, use //@version=6 and cite the documentation URLs.\n\n"
        "Response guidelines:\n"
        "1. Include working code examples whenever possible\n"
        "2. Explain each part of the code clearly\n"
        "3. Highlight common pitfalls or best practices\n"
        "4. If unsure about something, be transparent about it\n"
        "5. Format code with proper syntax highlighting\n"
        "6. Reference specific Pine Script v6 functions and features accurately\n"
        "7. Provide clear explanations of complex concepts with analogies when helpful\n"
        "8. When appropriate, mention TradingView-specific context"
    ),
    # defer_loading=False  ← default; keeps the tool+instructions in every prompt.
    # Flip to True when you add more capabilities and want progressive disclosure.
)


@pinescript_rag.tool
async def retrieve(ctx: RunContext[Dependencies], search_query: str) -> str:
    """Retrieve relevant Pine Script documentation using hybrid search.

    Pipeline: vector + BM25 → RRF fusion → similarity threshold → MMR deduplication.

    Args:
        ctx: Run context with database pool and OpenAI client.
        search_query: The search query to find relevant documentation.

    Returns:
        Concatenated documentation snippets relevant to the query.
    """
    try:
        docs = await hybrid_retrieve(
            pool=ctx.deps.pool,
            openai_client=ctx.deps.openai,
            query=search_query,
            embedding_model=EMBEDDING_MODEL,
            candidates=RETRIEVAL_CANDIDATES,
            alpha=HYBRID_SEARCH_ALPHA,
            threshold=SIMILARITY_THRESHOLD,
            top_n=RERANK_TOP_N,
            mmr_lambda=MMR_LAMBDA,
            enable_rerank=False,  # off as in the evaluations: ms-marco hurts code docs
        )

        if not docs:
            return (
                "No relevant documentation found in the database. "
                "The database may need to be populated with Pine Script documentation."
            )

        logger.debug("Hybrid retrieval returned %d documents", len(docs))

        return "\n\n".join(
            f"# {doc.title}\nDocumentation URL: {doc.url}\n\n{doc.content}\n" for doc in docs
        )
    except Exception as e:
        logger.error("Error in retrieve tool: %s", e)
        return f"Error retrieving documentation: {e!s}"


# ---------------------------------------------------------------------------
# The Agent — thin shell composing capabilities
#
# v1: Agent(model, deps_type, result_type, system_prompt, model_settings)
#     + @agent.tool + @agent.system_prompt  (everything bolted on separately)
#
# v2: Agent(model, deps_type, output_type, capabilities=[...], model_settings)
#     Everything is a capability. The agent is just the composition root.
# ---------------------------------------------------------------------------

pinescript_agent = Agent(
    DEFAULT_MODEL,  # "openai:gpt-4o-mini"
    deps_type=Dependencies,
    output_type=PineScriptResult,  # v2 naming (was result_type in old v1)
    capabilities=[
        Thinking(effort=THINKING_EFFORT),  # extended reasoning, unified across providers
        pinescript_rag,  # retrieve tool + expert instructions
    ],
    model_settings={
        "temperature": LLM_TEMPERATURE,  # 0.2 — now actually applied every run
        "max_tokens": LLM_MAX_TOKENS,  # 2000
    },
)


# ---------------------------------------------------------------------------
# Database helper (unchanged)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def database_connect(create_db: bool = False) -> AsyncGenerator[asyncpg.Pool, None]:
    """Connect to the pgvector database."""
    db_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    logger.info("Connecting to database: %s", db_url)
    try:
        pool = await asyncpg.create_pool(db_url)
        logger.info("Database connection established")
        try:
            yield pool
        finally:
            await pool.close()
            logger.info("Database connection closed")
    except Exception as e:
        logger.error("Error connecting to database: %s", e)
        raise


# ---------------------------------------------------------------------------
# Model routing — v2 style
#
# v1: create_openrouter_model() → with agent.override(model=...): agent.run()
# v2: resolve_model() → agent.run(model=...)  (no context manager needed)
# ---------------------------------------------------------------------------


def resolve_model(preset: str | None = None) -> OpenAIChatModel | None:
    """Resolve a preset name or raw model ID to a Pydantic AI model.

    Returns None to use the agent's default model (OpenAI).
    """
    openrouter_api_key = configured_key("OPENROUTER_API_KEY")

    if preset:
        if preset in MODEL_PRESETS:
            cfg = get_preset(preset)
            model_id = str(cfg["model"])
            logger.info("Using preset '%s' → %s", preset, model_id)
        else:
            model_id = preset  # raw OpenRouter model ID
            logger.info("Using raw model ID: %s", model_id)

        if openrouter_api_key:
            provider = OpenAIProvider(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)
            return OpenAIChatModel(model_id, provider=provider)

        logger.warning("No OPENROUTER_API_KEY for preset '%s', using default", preset)
        return None

    # No preset: use OpenRouter default if key exists, else agent default
    if openrouter_api_key:
        provider = OpenAIProvider(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)
        return OpenAIChatModel(OPENROUTER_DEFAULT_MODEL, provider=provider)

    return None  # agent default (OpenAI)


# ---------------------------------------------------------------------------
# Run helper — now accepts message_history for proper multi-turn
# ---------------------------------------------------------------------------


async def run_agent(question: str, preset: str | None = None, message_history=None):
    """Run the agent with a specific question.

    Args:
        question: The user's Pine Script question.
        preset: Optional model preset name ('codex', 'opus', 'flash')
                or a raw OpenRouter model ID.
        message_history: Optional list[ModelMessage] from a previous run
                         (via result.new_messages()) for multi-turn context.
    """
    logger.info("Running agent with question: %s", question)

    openai_api_key = get_openai_api_key()
    openai = AsyncOpenAI(api_key=openai_api_key)

    try:
        async with database_connect(False) as pool:
            deps = Dependencies(openai=openai, pool=pool)
            model = resolve_model(preset)

            # v2: pass model= directly to run() — no override() context manager
            run_kwargs: dict = {"deps": deps}
            if model is not None:
                run_kwargs["model"] = model
                if preset in MODEL_PRESETS:
                    # a preset brings its own temperature and max tokens
                    cfg = get_preset(preset)
                    run_kwargs["model_settings"] = {
                        "temperature": float(cfg["temperature"]),
                        "max_tokens": int(cfg["max_tokens"]),
                    }
            if message_history:
                run_kwargs["message_history"] = message_history

            answer = await pinescript_agent.run(question, **run_kwargs)
            return answer

    except Exception as e:
        logger.error("Error running agent: %s", e)
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def main():
    """Run the agent from the command line."""
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "How do I define a variable in Pine Script v6?"

    print(f"Question: {question}")
    result = await run_agent(question)

    if result:
        print("\nResponse:")
        print(result.output.response)  # v2: .output (was .data)
        print(f"\nSnippets used: {result.output.snippets_used}")
    else:
        print("No response received from the agent.")


if __name__ == "__main__":
    asyncio.run(main())
