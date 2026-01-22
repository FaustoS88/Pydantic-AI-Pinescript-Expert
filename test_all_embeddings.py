#!/usr/bin/env python3
"""
Test all embedding models to find their dimensions.
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

async def test_embedding_model(client, model_name):
    """Test a specific embedding model"""
    try:
        print(f"🧪 Testing {model_name}...")
        
        response = await client.embeddings.create(
            input="Test text for dimension checking",
            model=model_name
        )
        
        embedding = response.data[0].embedding
        dimensions = len(embedding)
        
        print(f"   ✅ {model_name}: {dimensions} dimensions")
        return dimensions
        
    except Exception as e:
        print(f"   ❌ {model_name}: Error - {e}")
        return None

async def test_all_embeddings():
    """Test all available embedding models"""
    
    # Get configuration
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    print(f"🔧 Testing embedding models on: {base_url}")
    
    # Create client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # List of embedding models to test
    embedding_models = [
        "text-embedding-nomic-embed-text-v2-moe",
        "text-embedding-nomic-embed-text-v1.5@f16", 
        "text-embedding-nomic-ai-nomic-embed-text-v2-moe",
        "text-embedding-nomic-embed-text-v1.5@q4_k_m",
        "text-embedding-bge-reranker-v2-m3"
    ]
    
    print(f"\n📊 Testing {len(embedding_models)} embedding models:")
    
    results = {}
    for model in embedding_models:
        dimensions = await test_embedding_model(client, model)
        if dimensions:
            results[model] = dimensions
    
    print(f"\n📋 Summary:")
    models_1536 = []
    models_768 = []
    models_other = []
    
    for model, dims in results.items():
        if dims == 1536:
            models_1536.append(model)
            print(f"   ✅ {model}: {dims} dimensions (matches database)")
        elif dims == 768:
            models_768.append(model)
            print(f"   ⚠️  {model}: {dims} dimensions (needs schema change)")
        else:
            models_other.append(model)
            print(f"   ❓ {model}: {dims} dimensions (unexpected)")
    
    print(f"\n🎯 Recommendations:")
    if models_1536:
        print(f"   Use one of these models (1536 dimensions, matches current database):")
        for model in models_1536:
            print(f"     - {model}")
    elif models_768:
        print(f"   Option 1: Change database schema to use 768 dimensions")
        print(f"   Option 2: Use a different embedding model that produces 1536 dimensions")
        print(f"   Available 768-dimension models:")
        for model in models_768:
            print(f"     - {model}")
    else:
        print(f"   No suitable embedding models found")

if __name__ == "__main__":
    asyncio.run(test_all_embeddings())