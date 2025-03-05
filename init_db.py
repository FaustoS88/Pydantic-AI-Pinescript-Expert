#!/usr/bin/env python3
"""
Initialize the database for the PineScript Expert Agent

This script:
1. Creates the vector extension if not already installed
2. Creates the pinescript_docs table and index
3. Verifies the database setup

Run this script before running the crawler or using the agent.
"""

import os
import asyncio
import asyncpg
from dotenv import load_dotenv

from db_schema import DB_SCHEMA

# Load environment variables
load_dotenv()

async def init_database():
    """Initialize the database"""
    # Get database connection parameters
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
    
    print(f"Connecting to database: {db_url}")
    
    # Connect to the database
    try:
        conn = await asyncpg.connect(db_url)
        print("Connected to database")
        
        # Create schema
        print("Creating schema...")
        await conn.execute(DB_SCHEMA)
        print("Schema created successfully")
        
        # Verify tables
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        print("\nTables in database:")
        for table in tables:
            print(f"  - {table['table_name']}")
        
        # Verify vector extension
        extensions = await conn.fetch(
            "SELECT extname FROM pg_extension"
        )
        print("\nExtensions in database:")
        for ext in extensions:
            print(f"  - {ext['extname']}")
        
        # Check if vector extension is installed
        vector_ext = await conn.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        if vector_ext:
            print("\n✅ Vector extension is installed")
        else:
            print("\n❌ Vector extension is NOT installed")
            print("Please install the pgvector extension")
            
        # Done
        print("\nDatabase initialization complete")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        if 'conn' in locals():
            await conn.close()
            print("Database connection closed")

def main():
    """Main entry point"""
    asyncio.run(init_database())

if __name__ == "__main__":
    main()
