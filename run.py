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
    print("  python run.py interactive [--model PRESET]  - Run the interactive shell")
    print("  python run.py query <query> [--model PRESET] - Run a single query")
    print("  python run.py populate <docs_dir>            - Populate the database with documentation")
    print("  python run.py check                          - Check the database setup")
    print()
    print("Model presets (via --model):")
    print("  default  - openai/gpt-4.1-mini")
    print("  codex    - openai/gpt-5.3-codex")
    print("  opus     - anthropic/claude-opus-4-6")
    print("  flash    - google/gemini-3-flash-preview")
    print("  <any>    - pass a raw OpenRouter model ID directly")

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

def _extract_model_flag(args: list[str]) -> tuple[str | None, list[str]]:
    """Pull --model VALUE out of args, return (preset, remaining_args)."""
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            preset = args[idx + 1]
            remaining = args[:idx] + args[idx + 2:]
            return preset, remaining
        return None, args
    return None, args


async def run_command(command, args):
    """Run a command.

    Args:
        command: The command to run
        args: The arguments for the command
    """
    preset, args = _extract_model_flag(args)

    if command == "interactive":
        from interactive import main
        await main(preset=preset)

    elif command == "query":
        if not args:
            print("Error: No query provided")
            print("Usage: python run.py query <query> [--model PRESET]")
            return

        from agent import run_agent
        query = " ".join(args)
        if preset:
            from config import MODEL_PRESETS, get_preset
            label = preset if preset not in MODEL_PRESETS else f"{preset} ({get_preset(preset)['model']})"
            print(f"Model: {label}")
        result = await run_agent(query, preset=preset)

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
