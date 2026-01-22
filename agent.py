from __future__ import annotations

import asyncio
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

# Print API keys (masked) to debug
openai_key = os.getenv("OPENAI_API_KEY", "")
openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

print(f"OpenAI API key loaded: {openai_key[:4]}...{openai_key[-4:] if len(openai_key) > 8 else ''}")
print(f"OpenRouter API key loaded: {openrouter_key[:4]}...{openrouter_key[-4:] if len(openrouter_key) > 8 else ''}")
print(f"OpenAI model: {openai_model}")
print(f"OpenRouter model: {openrouter_model}")

def get_openai_api_key() -> str:
    """Get OpenAI API key with validation and user prompt if needed."""
    # Try to get from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if key is missing or has a placeholder value
    if not openai_api_key or openai_api_key in ["YOUR_OPENAI_API_KEY", "sk-...", "YOUR_OPE***_API"] or "YOUR_" in openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable is not set or has a placeholder value.")
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
            print(f"Updated OPENAI_API_KEY in {env_path} for future runs")
        except ImportError:
            print("Could not update .env file - dotenv.set_key not available")
            print(f"Please manually update {env_path} with your API key")
    
    # Verify it's not still using a placeholder
    if "YOUR_" in openai_api_key or openai_api_key == "sk-...":
        print("Warning: API key appears to be a placeholder value.")
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
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
pinescript_agent = Agent(
    f"openai:{openai_model}",  # Use configurable model
    deps_type=Dependencies,
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
        
        print(f"Generating embedding for query: {search_query}")
        embedding = await openai_client.embeddings.create(
            input=search_query,
            #model="text-embedding-3-small",
            model="text-embedding-nomic-embed-text-v1.5@f16",
        )
        print(f"Embedding generated successfully")

        embedding_vector = embedding.data[0].embedding
        embedding_json = pydantic_core.to_json(embedding_vector).decode()
        
        print(f"Querying database for relevant documentation")
        rows = await ctx.deps.pool.fetch(
            "SELECT url, title, content FROM pinescript_docs ORDER BY embedding <-> $1 LIMIT 8",
            embedding_json,
        )
        print(f"Found {len(rows)} relevant documentation snippets")
        
        if not rows:
            return "No relevant documentation found in the database. The database may need to be populated with Pine Script documentation."
        
        # Count the snippets for the result metadata
        ctx.custom_data = {"snippets_used": len(rows)}
        
        return "\n\n".join(
            f"# {row['title']}\nDocumentation URL: {row['url']}\n\n{row['content']}\n"
            for row in rows
        )
    except Exception as e:
        print(f"Error in retrieve tool: {e}")
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
    
    print(f"Connecting to database: {db_url}")
    try:
        pool = await asyncpg.create_pool(db_url)
        print("Database connection established")
        try:
            yield pool
        finally:
            await pool.close()
            print("Database connection closed")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

async def create_openrouter_model():
    """Create an OpenRouter model if the API key is available."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("OPENROUTER_API_KEY not found in environment variables, using OpenAI")
        return None
    
    # Get configurable model name
    openrouter_model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    
    # Create a custom model that uses OpenRouter
    class OpenRouterModel(OpenAIModel):
        def __init__(self, model_name=openrouter_model_name):
            super().__init__(
                model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key
            )
    
    # Use the configurable model ID
    return OpenRouterModel(openrouter_model_name)

async def run_agent(question: str):
    """Run the agent with a specific question."""
    print(f"Running agent with question: {question}")
    
    # Initialize OpenAI client with explicit key and configurable base_url
    openai_api_key = get_openai_api_key()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    openai = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_base_url
    )
    print(f"OpenAI client initialized with base_url: {openai_base_url}")
    
    # Check if OpenRouter should be used
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    use_openrouter = bool(openrouter_api_key)
    
    # Connect to database and run the agent
    try:
        async with database_connect(False) as pool:
            print("Database connection ready")
            
            # Create dependencies
            deps = Dependencies(
                openai=openai,
                pool=pool,
                openrouter_api_key=openrouter_api_key,
                use_openrouter=use_openrouter
            )
            
            # Override the model with OpenRouter if available
            if use_openrouter:
                openrouter_model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
                print(f"Using OpenRouter model: {openrouter_model_name}")
                try:
                    openrouter_model = await create_openrouter_model()
                    if openrouter_model:
                        with pinescript_agent.override(model=openrouter_model):
                            raw_result = await pinescript_agent.run(question, deps=deps)
                    else:
                        print("OpenRouter model creation failed, using default OpenAI model")
                        raw_result = await pinescript_agent.run(question, deps=deps)
                except Exception as e:
                    print(f"OpenRouter failed: {e}, falling back to OpenAI")
                    raw_result = await pinescript_agent.run(question, deps=deps)
            else:
                openai_model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                print(f"Using OpenAI model: {openai_model_name}")
                raw_result = await pinescript_agent.run(question, deps=deps)
                
            print("Agent response received")
            
            # Create a structured result
            snippets_used = 0
            if hasattr(raw_result, 'all_messages'):
                # Try to extract snippets_used from the context if available
                for message in raw_result.all_messages():
                    if hasattr(message, 'parts'):
                        for part in message.parts:
                            if hasattr(part, 'tool_calls'):
                                # This is a rough estimate - in practice you might want to track this differently
                                snippets_used = 8  # Default assumption
                                break
            
            # Create a mock result object that matches the expected structure
            class MockResult:
                def __init__(self, response, query, snippets):
                    self.data = PineScriptResult(
                        query=query,
                        response=response,
                        snippets_used=snippets
                    )
            
            # Extract the actual response text
            response_text = str(raw_result.data) if hasattr(raw_result, 'data') else str(raw_result)
            
            return MockResult(response_text, question, snippets_used)
            
    except Exception as e:
        print(f"Error running agent: {e}")
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