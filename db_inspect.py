#!/usr/bin/env python3
"""
Database Inspector for PineScript Expert Agent

This script provides utilities to inspect the database contents,
check vector quality, and perform test searches without running the full agent.
"""

import os
import sys
import asyncio
import json
import pydantic_core
from dotenv import load_dotenv
from agent import database_connect
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

async def count_entries():
    """Count the number of entries in the database"""
    async with database_connect(False) as pool:
        count = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
        print(f"Database contains {count} documentation sections")

async def list_titles(limit=20):
    """List document titles in the database"""
    async with database_connect(False) as pool:
        rows = await pool.fetch(
            "SELECT id, title, url FROM pinescript_docs ORDER BY id LIMIT $1",
            limit
        )
        
        print(f"Sample of {len(rows)} document titles:")
        for row in rows:
            print(f"[{row['id']}] {row['title']}")
            print(f"    URL: {row['url']}")
            print()

async def view_document(doc_id):
    """View a specific document by ID"""
    async with database_connect(False) as pool:
        row = await pool.fetchrow(
            "SELECT id, title, url, content FROM pinescript_docs WHERE id = $1",
            doc_id
        )
        
        if not row:
            print(f"No document found with ID {doc_id}")
            return
            
        print(f"Document #{row['id']}: {row['title']}")
        print(f"URL: {row['url']}")
        print("\nContent:")
        print("=" * 80)
        print(row['content'])
        print("=" * 80)

async def test_search(query):
    """Test search functionality"""
    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
        
    openai = AsyncOpenAI(api_key=openai_api_key)
    
    # Generate embedding for query
    print(f"Generating embedding for query: '{query}'")
    embedding = await openai.embeddings.create(
        input=query,
        model="text-embedding-3-small",
    )
    embedding_vector = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding_vector).decode()
    
    # Search database
    async with database_connect(False) as pool:
        print("Searching database...")
        rows = await pool.fetch(
            """
            SELECT id, url, title, content,
                   1 - (embedding <=> $1) AS similarity
            FROM pinescript_docs
            ORDER BY similarity DESC
            LIMIT 5
            """,
            embedding_json,
        )
        
        print(f"\nTop {len(rows)} results for '{query}':")
        print("=" * 80)
        for i, row in enumerate(rows):
            print(f"Result #{i+1} [Similarity: {row['similarity']:.4f}]")
            print(f"Title: {row['title']}")
            print(f"URL: {row['url']}")
            print(f"Content Preview: {row['content'][:150]}...")
            print("-" * 80)

async def verify_vectors():
    """Verify that all documents have valid embedding vectors"""
    async with database_connect(False) as pool:
        # Count total documents
        total = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
        
        # Check for NULL embeddings
        null_embeddings = await pool.fetchval(
            "SELECT COUNT(*) FROM pinescript_docs WHERE embedding IS NULL"
        )
        
        # Check for zero-length embeddings
        zero_length = await pool.fetchval(
            "SELECT COUNT(*) FROM pinescript_docs WHERE array_length(embedding, 1) = 0"
        )
        
        # Get dimension of embeddings
        dimension = await pool.fetchval(
            "SELECT array_length(embedding, 1) FROM pinescript_docs LIMIT 1"
        )
        
        print(f"Total documents: {total}")
        print(f"Documents with NULL embeddings: {null_embeddings}")
        print(f"Documents with zero-length embeddings: {zero_length}")
        print(f"Embedding dimension: {dimension}")
        
        if null_embeddings > 0 or zero_length > 0:
            print("\nWarning: Some documents have invalid embeddings!")
            print("This may affect search quality. Consider regenerating embeddings.")

async def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python db_inspect.py count - Count database entries")
        print("  python db_inspect.py list [limit] - List document titles")
        print("  python db_inspect.py view <id> - View document by ID")
        print("  python db_inspect.py search <query> - Test search functionality")
        print("  python db_inspect.py verify - Verify vector quality")
        return
    
    command = sys.argv[1]
    
    if command == "count":
        await count_entries()
    elif command == "list":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        await list_titles(limit)
    elif command == "view":
        if len(sys.argv) < 3:
            print("Error: Document ID required")
            return
        await view_document(int(sys.argv[2]))
    elif command == "search":
        if len(sys.argv) < 3:
            print("Error: Search query required")
            return
        await test_search(" ".join(sys.argv[2:]))
    elif command == "verify":
        await verify_vectors()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    asyncio.run(main())
