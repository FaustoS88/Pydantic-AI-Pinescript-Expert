#!/usr/bin/env python3
"""
List available models from the local server.
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

async def list_models():
    """List available models from the server"""
    
    # Get configuration
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    print(f"🔧 Configuration:")
    print(f"   API Key: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")
    print(f"   Base URL: {base_url}")
    
    # Create client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    try:
        print(f"\n📋 Listing available models...")
        
        # List models
        models = await client.models.list()
        
        print(f"\n✅ Found {len(models.data)} models:")
        for model in models.data:
            print(f"   - {model.id}")
            if hasattr(model, 'object'):
                print(f"     Type: {model.object}")
            if hasattr(model, 'owned_by'):
                print(f"     Owner: {model.owned_by}")
            print()
        
        # Look for embedding models specifically
        embedding_models = [m for m in models.data if 'embed' in m.id.lower()]
        if embedding_models:
            print(f"🎯 Embedding models found:")
            for model in embedding_models:
                print(f"   - {model.id}")
        else:
            print(f"⚠️  No embedding models found with 'embed' in the name")
        
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    asyncio.run(list_models())