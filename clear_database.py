#!/usr/bin/env python3
"""
Clear Pine Script Vector Database

This script clears all entries from the pinescript_docs table
to prepare for a fresh crawl without conflicting data.
"""

import asyncio
import os
from dotenv import load_dotenv
from agent import database_connect
from db_schema import validate_schema

# Load environment variables
load_dotenv()

async def clear_database():
    """Clear all records from the pinescript_docs table"""
    print("\n===== DATABASE CLEANUP UTILITY =====\n")
    
    # Verify database connection and structure
    async with database_connect(False) as pool:
        # Validate schema first
        schema_valid = await validate_schema(pool)
        if not schema_valid:
            print("Database schema is not valid, please run init_db.py first.")
            return False
            
        # Count total documents
        count = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
        
        if count == 0:
            print("Database is already empty.")
            return True
            
        print(f"Found {count} records in the database.")
        print("This operation will delete ALL records from the pinescript_docs table.")
        
        # Get user confirmation
        confirmation = input("Type 'DELETE' to confirm deletion: ")
        
        if confirmation.upper() != 'DELETE':
            print("Operation cancelled.")
            return False
            
        # Execute deletion
        try:
            await pool.execute("DELETE FROM pinescript_docs")
            
            # Verify deletion
            new_count = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
            
            print(f"Successfully deleted {count} records.")
            print(f"Database now contains {new_count} records.")
            
            return True
        except Exception as e:
            print(f"Error deleting records: {e}")
            return False

async def main():
    """Main function"""
    success = await clear_database()
    
    if success:
        print("\nDatabase cleared successfully. Ready for a fresh crawl.")
    else:
        print("\nFailed to clear database. Please check the error messages.")

if __name__ == "__main__":
    asyncio.run(main())