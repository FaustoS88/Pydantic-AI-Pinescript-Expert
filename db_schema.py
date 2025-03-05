"""
Database schema for Pine Script documentation.
This file contains the SQL to create the necessary tables and indexes.
"""

# SQL to create the database schema with pgvector extension
DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pinescript_docs (
  id serial PRIMARY KEY,
  url text NOT NULL UNIQUE,
  title text NOT NULL,
  content text NOT NULL,
  embedding vector(1536) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pinescript_docs_embedding ON pinescript_docs USING hnsw (embedding vector_l2_ops);
"""

# Function to validate the schema (can be called to ensure DB is ready)
async def validate_schema(pool):
    """Validate that the database schema is correctly set up.
    
    Args:
        pool: Database connection pool
        
    Returns:
        bool: Whether the schema is valid
    """
    try:
        # Check if pgvector extension is installed
        extension_exists = await pool.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        if not extension_exists:
            print("Error: pgvector extension is not installed")
            return False
            
        # Check if the pinescript_docs table exists
        table_exists = await pool.fetchval(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'pinescript_docs'"
        )
        if not table_exists:
            print("Error: pinescript_docs table does not exist")
            return False
            
        # Check if the embedding index exists
        index_exists = await pool.fetchval(
            "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_pinescript_docs_embedding'"
        )
        if not index_exists:
            print("Warning: idx_pinescript_docs_embedding index does not exist")
            # Not returning False here as missing index is not critical
            
        return True
    except Exception as e:
        print(f"Error validating schema: {e}")
        return False

# Function to create the schema
async def create_schema(pool):
    """Create the database schema.
    
    Args:
        pool: Database connection pool
        
    Returns:
        bool: Whether the schema was created successfully
    """
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(DB_SCHEMA)
        print("Database schema created successfully")
        return True
    except Exception as e:
        print(f"Error creating schema: {e}")
        return False
