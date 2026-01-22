#!/usr/bin/env python3
"""
Fix embedding dimensions mismatch.

This script helps fix issues when the embedding model changes and 
the database contains embeddings with different dimensions.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from agent import database_connect

# Load environment variables
load_dotenv()

async def check_embedding_dimensions():
    """Check the current embedding dimensions in the database"""
    async with database_connect(False) as pool:
        # Check if table exists
        table_exists = await pool.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'pinescript_docs')"
        )
        
        if not table_exists:
            print("❌ Table 'pinescript_docs' does not exist")
            return None
        
        # Count total documents
        total_docs = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
        print(f"📊 Total documents in database: {total_docs}")
        
        if total_docs == 0:
            print("📝 Database is empty - no embedding dimension issues")
            return None
        
        # Check embedding dimensions
        sample_embedding = await pool.fetchval(
            "SELECT embedding FROM pinescript_docs WHERE embedding IS NOT NULL LIMIT 1"
        )
        
        if sample_embedding is None:
            print("⚠️  No embeddings found in database")
            return None
        
        dimensions = len(sample_embedding)
        print(f"🔍 Current embedding dimensions in database: {dimensions}")
        
        # Check what the current model should produce
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        print(f"🤖 Current embedding model: {embedding_model}")
        
        expected_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "text-embedding-nomic-embed-text-v2-moe": 768,
            "text-embedding-nomic-embed-text-v1.5@f16": 768,
            "text-embedding-nomic-ai-nomic-embed-text-v2-moe": 768,
            "text-embedding-nomic-embed-text-v1.5@q4_k_m": 768,
            "text-embedding-bge-reranker-v2-m3": 1024,
        }
        
        expected = expected_dimensions.get(embedding_model, "unknown")
        print(f"📏 Expected dimensions for {embedding_model}: {expected}")
        
        if expected != "unknown" and dimensions != expected:
            print(f"❌ DIMENSION MISMATCH: Database has {dimensions}, but model expects {expected}")
            print("\n💡 Solutions:")
            print("1. Clear database and repopulate with correct model:")
            print("   python fix_embeddings.py --clear-and-repopulate")
            print("2. Change embedding model to match database:")
            print(f"   Update OPENAI_EMBEDDING_MODEL in .env to a model that produces {dimensions} dimensions")
            return False
        else:
            print("✅ Embedding dimensions match expected model")
            return True

async def clear_and_repopulate():
    """Clear the database and suggest repopulation"""
    print("🗑️  Clearing database...")
    
    async with database_connect(False) as pool:
        # Count before deletion
        count_before = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
        print(f"📊 Found {count_before} existing records")
        
        if count_before == 0:
            print("📝 Database is already empty")
            return
        
        # Confirm deletion
        response = input(f"Are you sure you want to delete {count_before} records? (y/N): ").lower().strip()
        if response not in ['y', 'yes']:
            print("❌ Operation cancelled")
            return
        
        # Delete all records
        await pool.execute("DELETE FROM pinescript_docs")
        
        # Count after deletion
        count_after = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
        print(f"✅ Deleted {count_before - count_after} records")
        print(f"📊 Database now contains {count_after} records")
        
        print("\n📚 To repopulate the database:")
        print("   python pinescript_crawler.py")

async def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--clear-and-repopulate":
        await clear_and_repopulate()
    else:
        print("🔍 Checking embedding dimensions...")
        result = await check_embedding_dimensions()
        
        if result is False:
            print("\n⚠️  Embedding dimension mismatch detected!")
            print("Run with --clear-and-repopulate to fix, or update your .env file")
            sys.exit(1)
        elif result is True:
            print("\n✅ Everything looks good!")
        else:
            print("\n📝 No issues detected (database empty or no embeddings)")

if __name__ == "__main__":
    asyncio.run(main())