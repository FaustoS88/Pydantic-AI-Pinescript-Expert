#!/usr/bin/env python3
"""
Interactive shell for the Pine Script Expert Agent — Pydantic AI v2.

Now uses message_history for proper multi-turn context: the model sees
its own prior answers when you ask follow-up questions.
"""

import asyncio
import sys
from dotenv import load_dotenv

from agent import run_agent, PineScriptResult, get_openai_api_key

load_dotenv(override=True)


def ensure_api_key():
    """Make sure we have a valid API key."""
    try:
        api_key = get_openai_api_key()
        print(f"Using OpenAI API key: {api_key[:4]}...{api_key[-4:]}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


class InteractiveAgent:
    """Interactive shell for the Pine Script Expert Agent."""

    def __init__(self, preset: str | None = None):
        self.message_history = []  # list[ModelMessage] from result.new_messages()
        self.running = True
        self.preset = preset
        self._print_welcome()

    def _print_welcome(self):
        print("\n" + "=" * 80)
        print("Pine Script Expert Agent - Interactive Shell".center(80))
        print("=" * 80)
        print("Ask any question about Pine Script v6 or type 'exit' to quit.")
        print("Type 'clear' to clear the conversation history.")
        if self.preset:
            print(f"Model preset: {self.preset}")
        print("=" * 80 + "\n")

    async def process_input(self, user_input: str) -> bool:
        """Process user input and return whether to continue."""
        if user_input.lower() in ("exit", "quit", "q"):
            print("\nExiting interactive shell...")
            return False

        if user_input.lower() in ("clear", "cls"):
            self.message_history = []
            print("\nConversation history cleared.")
            return True

        print("\nProcessing your question...")
        try:
            # v2: pass accumulated message_history for multi-turn context
            result = await run_agent(
                user_input,
                preset=self.preset,
                message_history=self.message_history or None,
            )

            if result and isinstance(result.output, PineScriptResult):
                print("\n" + "=" * 80)
                print(result.output.response)           # v2: .output (was .data)
                print("=" * 80)

                # Accumulate history so the next turn sees all prior context
                self.message_history = result.all_messages()

            else:
                print("\nError: Failed to get a response from the agent.")
        except Exception as e:
            print(f"\nError running agent: {e}")

        return True

    async def run(self):
        """Run the interactive shell."""
        while self.running:
            try:
                user_input = input("\n> ")
                if not user_input.strip():
                    continue
                self.running = await self.process_input(user_input)
            except KeyboardInterrupt:
                print("\nExiting interactive shell...")
                self.running = False
            except Exception as e:
                print(f"\nError: {e}")


async def main(preset: str | None = None):
    """Main function to run the interactive shell."""
    ensure_api_key()
    interactive = InteractiveAgent(preset=preset)
    await interactive.run()


if __name__ == "__main__":
    asyncio.run(main())