#!/usr/bin/env python3
"""
Interactive shell for the Pine Script Expert Agent.

Run this script to have an interactive conversation with the agent.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

from agent import run_agent, PineScriptResult, get_openai_api_key

# Load environment variables
load_dotenv(override=True)

# Make sure we have a valid API key before starting
def ensure_api_key():
    """Make sure we have a valid API key"""
    try:
        api_key = get_openai_api_key()
        print(f"Using OpenAI API key: {api_key[:4]}...{api_key[-4:]}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

class InteractiveAgent:
    """Interactive shell for the Pine Script Expert Agent"""
    
    def __init__(self):
        self.message_history = []
        self.running = True
        self.print_welcome()
        
    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "=" * 80)
        print("Pine Script Expert Agent - Interactive Shell".center(80))
        print("=" * 80)
        print("Ask any question about Pine Script v6 or type 'exit' to quit.")
        print("Type 'clear' to clear the conversation history.")
        print("=" * 80 + "\n")
        
    async def process_input(self, user_input: str) -> bool:
        """Process user input.
        
        Args:
            user_input: The user's input
            
        Returns:
            bool: Whether to continue the conversation
        """
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nExiting interactive shell...")
            return False
            
        if user_input.lower() in ['clear', 'cls']:
            self.message_history = []
            print("\nConversation history cleared.")
            return True
            
        print("\nProcessing your question...")
        try:
            result = await run_agent(user_input)
            
            if result and isinstance(result.data, PineScriptResult):
                print("\n" + "=" * 80)
                print(result.data.response)
                print("=" * 80)
                
                # Add to message history for potential future use
                self.message_history.append({
                    "query": user_input,
                    "response": result.data.response
                })
            else:
                print("\nError: Failed to get a response from the agent.")
        except Exception as e:
            print(f"\nError running agent: {e}")
            
        return True
        
    async def run(self):
        """Run the interactive shell"""
        while self.running:
            try:
                # Get user input
                user_input = input("\n> ")
                
                # Skip empty input
                if not user_input.strip():
                    continue
                    
                # Process input
                self.running = await self.process_input(user_input)
                
            except KeyboardInterrupt:
                print("\nExiting interactive shell...")
                self.running = False
            except Exception as e:
                print(f"\nError: {e}")

async def main():
    """Main function to run the interactive shell"""
    # Ensure we have a valid API key before starting
    ensure_api_key()
    
    # Run the interactive shell
    interactive = InteractiveAgent()
    await interactive.run()

if __name__ == "__main__":
    asyncio.run(main())