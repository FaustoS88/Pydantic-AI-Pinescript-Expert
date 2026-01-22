#!/usr/bin/env python3
"""
Migrate database from 1536 to 768 dimensions.
This script will drop the existing table and recreate it with 768 dimensions.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from agent import database_connect

# Load environment variables
load_dotenv()

async def migrate_database():
    """Migrate the database to use 768 dimensions"""
    
    print("🔄 Migrating database to use 768-dimensional embeddings...")
    
    async with database_connect(False) as pool:
        try:
            # Check if table exists and get current schema
            table_info = await pool.fetchrow("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = 'pinescript_docs' AND column_name = 'embedding'
            """)
            
            if table_info:
                print(f"📊 Current table found with embedding column: {table_info['data_type']}")
                
                # Count existing records
                count = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
                print(f"📋 Found {count} existing records")
                
                if count > 0:
                    response = input(f"⚠️  This will delete {count} existing records. Continue? (y/N): ").lower().strip()
                    if response not in ['y', 'yes']:
                        print("❌ Migration cancelled")
                        return False
            else:
                print("📝 No existing table found")
            
            # Drop existing table and recreate with new schema
            print("🗑️  Dropping existing table...")
            await pool.execute("DROP TABLE IF EXISTS pinescript_docs CASCADE")
            
            print("🔧 Creating new table with 768 dimensions...")
            await pool.execute("""
                CREATE TABLE pinescript_docs (
                  id serial PRIMARY KEY,
                  url text NOT NULL UNIQUE,
                  title text NOT NULL,
                  content text NOT NULL,
                  embedding vector(768) NOT NULL
                );
            """)
            
            print("📊 Creating index...")
            await pool.execute("""
                CREATE INDEX idx_pinescript_docs_embedding 
                ON pinescript_docs USING hnsw (embedding vector_l2_ops);
            """)
            
            print("✅ Migration completed successfully!")
            print("📚 You can now run the crawler to populate the database:")
            print("   python pinescript_crawler.py")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during migration: {e}")
            return False

async def main():
    """Main function"""
    print("🔍 Database Migration Tool")
    print("This will migrate your database to use 768-dimensional embeddings")
    print("to match your available embedding models.\n")
    
    success = await migrate_database()
    
    if success:
        print("\n🎉 Migration completed!")
        print("Your database is now ready for 768-dimensional embeddings.")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())