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
  embedding vector(1536) NOT NULL,
  search_vector tsvector,
  chunk_index       INTEGER DEFAULT 0,
  content_type      TEXT    DEFAULT 'reference',
  contextual_prefix TEXT
);

-- Use cosine ops (not L2) — OpenAI embeddings are normalized for cosine similarity
CREATE INDEX IF NOT EXISTS idx_pinescript_docs_embedding
  ON pinescript_docs USING hnsw (embedding vector_cosine_ops);

-- GIN index for full-text search (BM25-style)
CREATE INDEX IF NOT EXISTS idx_pinescript_docs_search
  ON pinescript_docs USING gin (search_vector);
"""

# Migration SQL to upgrade existing tables with tsvector + cosine index
DB_MIGRATION = """
-- Add search_vector column if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'pinescript_docs' AND column_name = 'search_vector'
  ) THEN
    ALTER TABLE pinescript_docs ADD COLUMN search_vector tsvector;
  END IF;
END $$;

-- Populate search_vector from existing title + content
UPDATE pinescript_docs
SET search_vector = to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
WHERE search_vector IS NULL;

-- Create GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_pinescript_docs_search
  ON pinescript_docs USING gin (search_vector);

-- Recreate HNSW index with cosine ops (drop L2 ops index first if exists)
DROP INDEX IF EXISTS idx_pinescript_docs_embedding;
CREATE INDEX IF NOT EXISTS idx_pinescript_docs_embedding
  ON pinescript_docs USING hnsw (embedding vector_cosine_ops);

-- Tier 2: metadata columns (idempotent)
ALTER TABLE pinescript_docs ADD COLUMN IF NOT EXISTS chunk_index       INTEGER DEFAULT 0;
ALTER TABLE pinescript_docs ADD COLUMN IF NOT EXISTS content_type      TEXT    DEFAULT 'reference';
ALTER TABLE pinescript_docs ADD COLUMN IF NOT EXISTS contextual_prefix TEXT;
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


async def run_migration(pool):
    """Run migration to add tsvector column and update indexes.

    Safe to run multiple times — all operations are idempotent.

    Args:
        pool: Database connection pool

    Returns:
        bool: Whether the migration ran successfully
    """
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(DB_MIGRATION)
        print("Migration completed successfully")
        return True
    except Exception as e:
        print(f"Error running migration: {e}")
        return False
