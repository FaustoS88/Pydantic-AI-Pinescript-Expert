#!/usr/bin/env python3
"""
OpenAI API Key Debugging Script

This script helps debug issues with the OpenAI API key by testing
different aspects of the initialization and request process.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent

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
        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=openai_base_url
        )
        models = await async_client.models.list()
        print("✅ Async OpenAI client works!")
        print(f"Found {len(models.data)} models")
        print(f"Using base_url: {openai_base_url}")
    except Exception as e:
        print(f"❌ Async OpenAI client failed: {e}")
    
    # Test Pydantic AI OpenAIModel
    print_section("Pydantic AI OpenAIModel Test")
    try:
        openai_model = OpenAIModel("gpt-4o", api_key=api_key)
        print("✅ OpenAIModel initialized successfully!")
        print(f"Model name: {openai_model.model_name}")
        
        # Test making a request with the model
        response = await openai_model("Hello, world!")
        print("✅ Model request successful!")
        print(f"Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ OpenAIModel failed: {e}")
    
    # Test Pydantic AI Agent
    print_section("Pydantic AI Agent Test")
    try:
        # Try creating a minimal agent
        agent = Agent(
            model="openai:gpt-4o",
            model_settings={
                "api_key": api_key
            }
        )
        print("✅ Agent created successfully!")
        
        # Print the agent's model settings
        print("Agent model settings:")
        if hasattr(agent, 'model_settings'):
            for key, value in agent.model_settings.items():
                if key == "api_key" and value:
                    print(f"  api_key: {value[:4]}...{value[-4:]}")
                else:
                    print(f"  {key}: {value}")
        
        # Try making a request with the agent
        try:
            result = agent.run_sync("Hello, world!", model_settings={"api_key": api_key})
            print("✅ Agent request successful!")
            print(f"Response: {result.data[:50]}...")
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