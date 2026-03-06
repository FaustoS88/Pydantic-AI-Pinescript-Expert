from __future__ import annotations

import asyncio
import logging
import os
import sys
import pydantic_core
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

import asyncpg
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv

from config import (
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OPENROUTER_BASE_URL,
    OPENROUTER_DEFAULT_MODEL,
    VECTOR_SEARCH_LIMIT,
    DEFAULT_DATABASE_URL,
)

# Force reload environment variables
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Log masked API keys at DEBUG level for startup diagnostics
openai_key = os.getenv("OPENAI_API_KEY", "")
openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
logger.debug(
    "OpenAI API key loaded: %s...%s",
    openai_key[:4],
    openai_key[-4:] if len(openai_key) > 8 else "",
)
logger.debug(
    "OpenRouter API key loaded: %s...%s",
    openrouter_key[:4],
    openrouter_key[-4:] if len(openrouter_key) > 8 else "",
)

def get_openai_api_key() -> str:
    """Get OpenAI API key with validation and user prompt if needed."""
    # Try to get from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check if key is missing or has a placeholder value
    if not openai_api_key or openai_api_key in ["YOUR_OPENAI_API_KEY", "sk-...", "YOUR_OPE***_API"] or "YOUR_" in openai_api_key:
        logger.warning("OPENAI_API_KEY is not set or has a placeholder value.")
        print("Please enter your OpenAI API key:")
        openai_api_key = input("> ")

        if not openai_api_key:
            raise ValueError("OpenAI API key is required to continue.")

        # Save to environment for this session
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Also update the .env file for future runs
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        try:
            from dotenv import set_key
            set_key(env_path, "OPENAI_API_KEY", openai_api_key)
            logger.info("Updated OPENAI_API_KEY in %s for future runs", env_path)
        except ImportError:
            logger.warning("Could not update .env file - dotenv.set_key not available")
            logger.warning("Please manually update %s with your API key", env_path)

    # Verify it's not still using a placeholder
    if "YOUR_" in openai_api_key or openai_api_key == "sk-...":
        logger.warning("API key appears to be a placeholder value.")
        print("Please enter your actual OpenAI API key:")
        openai_api_key = input("> ")

        if not openai_api_key:
            raise ValueError("OpenAI API key is required to continue.")

        # Save to environment for this session
        os.environ["OPENAI_API_KEY"] = openai_api_key

    return openai_api_key


# Define result structure for better tracking
class PineScriptResult(BaseModel):
    query: str = Field(description="The original query")
    response: str = Field(description="The generated response")
    snippets_used: int = Field(description="Number of documentation snippets used")

@dataclass
class Dependencies:
    """Dependencies for the PineScript Expert Agent"""
    openai: AsyncOpenAI
    pool: asyncpg.Pool
    openrouter_api_key: str = None
    use_openrouter: bool = False

# Initialize the agent with appropriate settings
pinescript_agent = Agent(
    DEFAULT_MODEL,
    deps_type=Dependencies,
    result_type=PineScriptResult,
    system_prompt=(
        "You are a Pine Script v6 expert assistant. Pine Script is the programming language used in TradingView "
        "for creating custom indicators and strategies for technical analysis of financial markets. "
        "Your task is to provide clear, accurate information about Pine Script v6 based on the official documentation. "
        "Always include code examples in your explanations when relevant. "
        "Focus on being practical and giving working solutions for user problems."
    ),
    model_settings={
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
)

@pinescript_agent.system_prompt
async def add_style_prompt(ctx: RunContext[Dependencies]) -> str:
    """Add a style prompt for consistent responses"""
    return (
        "When answering questions about Pine Script, follow these guidelines:\n"
        "1. Include working code examples whenever possible\n"
        "2. Explain each part of the code clearly\n"
        "3. Highlight any common pitfalls or best practices\n"
        "4. If you're unsure about something, be transparent about it\n"
        "5. Format your code with proper syntax highlighting\n"
        "6. When appropriate, mention TradingView-specific context\n"
        "7. Reference specific Pine Script v6 functions and features accurately\n"
        "8. Provide clear explanations of complex concepts with analogies when helpful\n"
    )

@pinescript_agent.tool
async def retrieve(ctx: RunContext[Dependencies], search_query: str) -> str:
    """Retrieve relevant Pine Script documentation based on a search query.

    Args:
        ctx: The run context with dependencies
        search_query: The search query to find relevant documentation

    Returns:
        str: Concatenated documentation snippets relevant to the query
    """
    try:
        # Always use OpenAI for embeddings, never OpenRouter
        # This is the critical fix to ensure embeddings always use OpenAI
        openai_client = ctx.deps.openai

        logger.debug("Generating embedding for query: %s", search_query)
        embedding = await openai_client.embeddings.create(
            input=search_query,
            model=EMBEDDING_MODEL,
        )
        logger.debug("Embedding generated successfully")

        embedding_vector = embedding.data[0].embedding
        embedding_json = pydantic_core.to_json(embedding_vector).decode()

        logger.debug("Querying database for relevant documentation")
        rows = await ctx.deps.pool.fetch(
            f"SELECT url, title, content FROM pinescript_docs ORDER BY embedding <-> $1 LIMIT {VECTOR_SEARCH_LIMIT}",
            embedding_json,
        )
        logger.debug("Found %d relevant documentation snippets", len(rows))

        if not rows:
            return "No relevant documentation found in the database. The database may need to be populated with Pine Script documentation."

        # Count the snippets for the result metadata
        ctx.custom_data = {"snippets_used": len(rows)}

        return "\n\n".join(
            f"# {row['title']}\nDocumentation URL: {row['url']}\n\n{row['content']}\n"
            for row in rows
        )
    except Exception as e:
        logger.error("Error in retrieve tool: %s", e)
        return f"Error retrieving documentation: {str(e)}"

@asynccontextmanager
async def database_connect(create_db: bool = False) -> AsyncGenerator[asyncpg.Pool, None]:
    """Connect to the database with the Pine Script documentation.

    Args:
        create_db: Whether to create the database if it doesn't exist

    Yields:
        asyncpg.Pool: A connection pool to the database
    """
    # Use the connection string directly from environment variables
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

def create_openrouter_model(model_id: str | None = None):
    """Create an OpenRouter model.

    Args:
        model_id: OpenRouter model ID (e.g. ``openai/gpt-5.3-codex``).
                  Falls back to OPENROUTER_DEFAULT_MODEL from config.
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        logger.info("OPENROUTER_API_KEY not found, using OpenAI")
        return None

    chosen = model_id or OPENROUTER_DEFAULT_MODEL
    logger.info("Creating OpenRouter model: %s", chosen)
    return OpenAIModel(chosen, base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)

async def run_agent(question: str, preset: str | None = None):
    """Run the agent with a specific question.

    Args:
        question: The user's Pine Script question.
        preset:   Optional model preset name (``codex``, ``opus``, ``flash``)
                  or a raw OpenRouter model ID (e.g. ``anthropic/claude-opus-4-6``).
                  When omitted the agent falls back to OpenRouter default or OpenAI.
    """
    logger.info("Running agent with question: %s", question)

    openai_api_key = get_openai_api_key()
    openai = AsyncOpenAI(api_key=openai_api_key)

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    use_openrouter = bool(openrouter_api_key)

    # Resolve model override from preset name or raw model ID
    model_override = None
    if preset:
        from config import MODEL_PRESETS, get_preset
        if preset in MODEL_PRESETS:
            cfg = get_preset(preset)
            model_override = create_openrouter_model(cfg["model"])
            logger.info("Using preset '%s' → %s", preset, cfg["model"])
        else:
            # Treat as a raw OpenRouter model ID
            model_override = create_openrouter_model(preset)
            logger.info("Using raw model ID: %s", preset)

    try:
        async with database_connect(False) as pool:
            deps = Dependencies(
                openai=openai,
                pool=pool,
                openrouter_api_key=openrouter_api_key,
                use_openrouter=use_openrouter,
            )

            if model_override:
                with pinescript_agent.override(model=model_override):
                    answer = await pinescript_agent.run(question, deps=deps)
            elif use_openrouter:
                logger.info("Using OpenRouter default model")
                or_model = create_openrouter_model()
                if or_model:
                    with pinescript_agent.override(model=or_model):
                        answer = await pinescript_agent.run(question, deps=deps)
                else:
                    answer = await pinescript_agent.run(question, deps=deps)
            else:
                logger.info("Using default OpenAI model")
                answer = await pinescript_agent.run(question, deps=deps)

            return answer
    except Exception as e:
        logger.error("Error running agent: %s", e)
        return None

async def main():
    """Main function to run the agent from the command line."""
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "How do I define a variable in Pine Script v6?"

    print(f"Question: {question}")
    result = await run_agent(question)

    if result:
        print("\nResponse:")
        print(result.data.response)
        print(f"\nSnippets used: {result.data.snippets_used}")
    else:
        print("No response received from the agent.")

if __name__ == "__main__":
    asyncio.run(main())
