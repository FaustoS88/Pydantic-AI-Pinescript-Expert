#!/usr/bin/env python3
"""
Initialize the database for the PineScript Expert Agent

This script:
1. Creates the vector extension if not already installed
2. Creates the pinescript_docs table and index
3. Verifies the database setup

Run this script before running the crawler or using the agent.
"""

import logging
import os
import asyncio
import asyncpg
from dotenv import load_dotenv

from db_schema import DB_SCHEMA

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

async def init_database():
    """Initialize the database"""
    # Get database connection parameters
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")

    logger.info("Connecting to database: %s", db_url)

    # Connect to the database
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("Connected to database")

        # Create schema
        logger.info("Creating schema...")
        await conn.execute(DB_SCHEMA)
        logger.info("Schema created successfully")

        # Verify tables
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        logger.info("Tables in database:")
        for table in tables:
            logger.info("  - %s", table['table_name'])

        # Verify vector extension
        extensions = await conn.fetch(
            "SELECT extname FROM pg_extension"
        )
        logger.info("Extensions in database:")
        for ext in extensions:
            logger.info("  - %s", ext['extname'])

        # Check if vector extension is installed
        vector_ext = await conn.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        if vector_ext:
            logger.info("Vector extension is installed")
        else:
            logger.warning("Vector extension is NOT installed — please install pgvector")

        # Done
        logger.info("Database initialization complete")

    except Exception as e:
        logger.error("Error initializing database: %s", e)
    finally:
        if 'conn' in locals():
            await conn.close()
            logger.info("Database connection closed")

def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(init_database())

if __name__ == "__main__":
    main()
