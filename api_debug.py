#!/usr/bin/env python3
"""
OpenAI API Key Debugging Script

This script helps debug issues with the OpenAI API key by testing
different aspects of the initialization and request process.
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load environment variables
load_dotenv(override=True)


def print_section(title):
    """Print a section title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


async def main():
    """Main function to debug API key issues"""
    print_section("Environment Variables")

    # Check if API key is set in environment
    api_key = os.getenv("OPENAI_API_KEY")
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if api_key and len(api_key) > 8 else "None"
    print(f"OPENAI_API_KEY from environment: {masked_key}")

    # Check for malformed placeholders
    if api_key in ["YOUR_OPENAI_API_KEY", "sk-...", "YOUR_OPE***_API"] or not api_key:
        print("WARNING: API key appears to be a placeholder or is missing!")

        # Prompt for key
        print("Enter your OpenAI API key for testing:")
        api_key = input("> ").strip()
        os.environ["OPENAI_API_KEY"] = api_key
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if api_key and len(api_key) > 8 else "None"
        print(f"Using API key: {masked_key}")

    # Test standard OpenAI client
    print_section("Standard OpenAI Client Test")
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        print("✅ Standard OpenAI client works!")
        print(f"Found {len(models.data)} models")
    except Exception as e:
        print(f"❌ Standard OpenAI client failed: {e}")

    # Test AsyncOpenAI client
    print_section("Async OpenAI Client Test")
    try:
        async_client = AsyncOpenAI(api_key=api_key)
        models = await async_client.models.list()
        print("✅ Async OpenAI client works!")
        print(f"Found {len(models.data)} models")
    except Exception as e:
        print(f"❌ Async OpenAI client failed: {e}")

    # Test Pydantic AI OpenAIChatModel
    # v2: the API key travels on the provider, not as a model kwarg.
    print_section("Pydantic AI OpenAIChatModel Test")
    openai_model = None
    try:
        openai_model = OpenAIChatModel("gpt-4o", provider=OpenAIProvider(api_key=api_key))
        print("✅ OpenAIChatModel initialized successfully!")
        print(f"Model name: {openai_model.model_name}")
    except Exception as e:
        print(f"❌ OpenAIChatModel failed: {e}")

    # Test Pydantic AI Agent
    print_section("Pydantic AI Agent Test")
    try:
        # v2: pass the configured model object; api_key is not a model_setting.
        agent = Agent(openai_model) if openai_model else Agent("openai:gpt-4o")
        print("✅ Agent created successfully!")

        # Try making a request with the agent
        try:
            result = agent.run_sync("Hello, world!")
            print("✅ Agent request successful!")
            print(f"Response: {str(result.output)[:50]}...")
        except Exception as e:
            print(f"❌ Agent request failed: {e}")

            # Check if the error is related to API key
            error_str = str(e).lower()
            if "api key" in error_str or "openai" in error_str:
                print("\nDetected potential API key issue in agent request.")
                print("Suggestions:")
                print("1. Try updating the pinescript_agent initialization in agent.py")
                print("2. Add model_settings with api_key explicitly when creating the agent")
                print("3. Modify the retrieve tool to use a separate OpenAI client")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")

    print_section("Summary")
    print("This debug information should help identify where the API key issue is occurring.")
    print("Look for any failures above and focus debugging on those components.")
    print("\nNext steps:")
    print("1. If all tests pass, the issue is likely in how your agent is using the API key")
    print("2. If some tests fail, focus on those specific components")
    print("3. Check if `model_settings` in the agent includes the correct API key")


if __name__ == "__main__":
    asyncio.run(main())
