from __future__ import annotations

import asyncio
import logging
import os
import sys
import json
import pydantic_core
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, AsyncGenerator

import asyncpg
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv

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
    "openai:gpt-4o-mini",  # Default model
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
        "temperature": 0.2,  # Lower temperature for more accurate, factual responses
        "max_tokens": 2000   # Allow for detailed responses with code examples
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
            model="text-embedding-3-small",
        )
        logger.debug("Embedding generated successfully")

        embedding_vector = embedding.data[0].embedding
        embedding_json = pydantic_core.to_json(embedding_vector).decode()

        logger.debug("Querying database for relevant documentation")
        rows = await ctx.deps.pool.fetch(
            "SELECT url, title, content FROM pinescript_docs ORDER BY embedding <-> $1 LIMIT 8",
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
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")

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

async def create_openrouter_model():
    """Create an OpenRouter model if the API key is available."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        logger.info("OPENROUTER_API_KEY not found in environment variables, using OpenAI")
        return None

    # Create a custom model that uses OpenRouter
    class OpenRouterModel(OpenAIModel):
        def __init__(self, model_name="openai/gpt-4.1-mini"):
            super().__init__(
                model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key
            )

    # Use a model ID that OpenRouter actually supports
    return OpenRouterModel("openai/gpt-4.1-mini")

async def run_agent(question: str):
    """Run the agent with a specific question."""
    logger.info("Running agent with question: %s", question)

    # Initialize OpenAI client with explicit key
    openai_api_key = get_openai_api_key()
    openai = AsyncOpenAI(api_key=openai_api_key)
    logger.debug("OpenAI client initialized")

    # Check if OpenRouter should be used
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    use_openrouter = bool(openrouter_api_key)

    # Connect to database and run the agent
    try:
        async with database_connect(False) as pool:
            logger.debug("Database connection ready")

            # Create dependencies
            deps = Dependencies(
                openai=openai,
                pool=pool,
                openrouter_api_key=openrouter_api_key,
                use_openrouter=use_openrouter
            )

            # Override the model with OpenRouter if available
            if use_openrouter:
                logger.info("Using OpenRouter for queries")
                try:
                    openrouter_model = await create_openrouter_model()
                    if openrouter_model:
                        with pinescript_agent.override(model=openrouter_model):
                            answer = await pinescript_agent.run(question, deps=deps)
                    else:
                        logger.warning("OpenRouter model creation failed, using default OpenAI model")
                        answer = await pinescript_agent.run(question, deps=deps)
                except Exception as e:
                    logger.warning("OpenRouter failed: %s, falling back to OpenAI", e)
                    answer = await pinescript_agent.run(question, deps=deps)
            else:
                logger.info("Using default OpenAI model")
                answer = await pinescript_agent.run(question, deps=deps)

            logger.debug("Agent response received")
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
