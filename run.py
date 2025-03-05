#!/usr/bin/env python3
"""
Run script for the Pine Script Expert Agent.
This script provides a convenient way to run different commands.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_usage():
    """Print usage instructions"""
    print("Usage:")
    print("  python run.py interactive   - Run the interactive shell")
    print("  python run.py query <query> - Run a single query")
    print("  python run.py populate <docs_dir> - Populate the database with documentation")
    print("  python run.py check - Check the database setup")

async def check_database():
    """Check the database setup"""
    from agent import database_connect
    from db_schema import validate_schema
    
    try:
        async with database_connect(False) as pool:
            is_valid = await validate_schema(pool)
            if is_valid:
                print("Database schema is valid")
                
                # Count documents
                count = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
                print(f"Database contains {count} documentation sections")
                
                if count == 0:
                    print("Warning: No documentation found in database. Run 'python run.py populate <docs_dir>' to add documentation.")
            else:
                print("Database schema is not valid. Run 'python run.py populate <docs_dir>' to set up the database.")
    except Exception as e:
        print(f"Error checking database: {e}")

async def run_command(command, args):
    """Run a command.
    
    Args:
        command: The command to run
        args: The arguments for the command
    """
    if command == "interactive":
        from interactive import main
        await main()
        
    elif command == "query":
        if not args:
            print("Error: No query provided")
            print("Usage: python run.py query <query>")
            return
            
        from agent import run_agent
        query = " ".join(args)
        result = await run_agent(query)
        
        if result:
            print("\nResponse:")
            print(result.data.response)
            
    elif command == "populate":
        if not args:
            print("Error: No docs directory provided")
            print("Usage: python run.py populate <docs_dir>")
            return
            
        docs_dir = args[0]
        if not os.path.exists(docs_dir):
            print(f"Error: Directory {docs_dir} does not exist")
            return
            
        from populate_db import build_search_db
        await build_search_db(docs_dir)
        
    elif command == "check":
        await check_database()
        
    else:
        print(f"Unknown command: {command}")
        print_usage()

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print_usage()
        return
        
    command = sys.argv[1]
    args = sys.argv[2:]
    
    asyncio.run(run_command(command, args))

if __name__ == "__main__":
    main()
