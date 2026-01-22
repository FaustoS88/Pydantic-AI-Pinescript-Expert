#!/usr/bin/env python3
"""
Test embedding dimensions to debug the issue.
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

async def test_embedding():
    """Test the embedding model and check dimensions"""
    
    # Get configuration
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5@f16")
    
    print(f"🔧 Configuration:")
    print(f"   API Key: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")
    print(f"   Base URL: {base_url}")
    print(f"   Embedding Model: {embedding_model}")
    
    # Create client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Test text
    test_text = "This is a test for Pine Script documentation embedding."
    
    try:
        print(f"\n🧪 Testing embedding generation...")
        print(f"   Text: {test_text}")
        
        # Generate embedding
        response = await client.embeddings.create(
            input=test_text,
            model=embedding_model
        )
        
        # Check dimensions
        embedding = response.data[0].embedding
        dimensions = len(embedding)
        
        print(f"\n✅ Embedding generated successfully!")
        print(f"   Dimensions: {dimensions}")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Check if dimensions match expectation
        if dimensions == 1536:
            print(f"✅ Dimensions match expected 1536!")
        elif dimensions == 768:
            print(f"⚠️  Dimensions are 768, but database expects 1536")
            print(f"   This suggests the model is producing different dimensions than expected")
        else:
            print(f"❓ Unexpected dimensions: {dimensions}")
        
        return dimensions
        
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_embedding())