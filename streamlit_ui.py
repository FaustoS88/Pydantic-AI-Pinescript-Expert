"""
Streamlit UI for the Pine Script Expert Agent — Pydantic AI v2.

Key fix over v1: message history now includes BOTH user and assistant
messages, so the model sees its own prior answers in multi-turn chats.
"""

import asyncio
import os
import pickle
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI

# v2 message types for proper history reconstruction
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from agent import Dependencies, database_connect, pinescript_agent, resolve_model  # noqa: E402
from config import configured_key  # noqa: E402
from db_schema import validate_schema  # noqa: E402

load_dotenv(override=True)

HISTORY_FILE = os.path.join(current_dir, "chat_history.pkl")


# ---------------------------------------------------------------------------
# Chat history persistence (pickle — swap for Harness Memory later)
# ---------------------------------------------------------------------------


def load_chat_history() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
    return []


def save_chat_history(messages: list[dict]) -> None:
    try:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(messages, f)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")


# ---------------------------------------------------------------------------
# THE FIX: convert chat history to model messages including assistant turns
#
# v1 only did:  ModelRequest(parts=[UserPromptPart(content=...)])
#               → model never saw its own prior answers
#
# v2 does both: ModelRequest  for user turns
#               ModelResponse for assistant turns
# ---------------------------------------------------------------------------


def build_message_history(messages: list[dict]) -> list:
    """Convert Streamlit chat dicts to Pydantic AI model messages."""
    model_messages = []
    for msg in messages:
        if msg["role"] == "user":
            model_messages.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
        elif msg["role"] == "assistant":
            model_messages.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
    return model_messages


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="PineScript Expert", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stChatMessage { padding: 0.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()


# ---------------------------------------------------------------------------
# Setup verification
# ---------------------------------------------------------------------------


async def verify_setup():
    openai_api_key = configured_key("OPENAI_API_KEY")
    if not openai_api_key:
        st.sidebar.error("⚠️ OpenAI API key not configured.")
        return False, None, 0

    try:
        async with database_connect(False) as pool:
            if await validate_schema(pool):
                doc_count = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
                return True, openai_api_key, doc_count
            st.sidebar.error("⚠️ Database schema invalid.")
            return False, openai_api_key, 0
    except Exception as e:
        st.sidebar.error(f"⚠️ Database connection error: {e}")
        return False, openai_api_key, 0


# ---------------------------------------------------------------------------
# Query processing — v2: passes full history (user + assistant) to the model
# ---------------------------------------------------------------------------


async def process_query(prompt: str, history: list[dict]):
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async with database_connect(False) as pool:
        deps = Dependencies(openai=openai_client, pool=pool)

        # Build proper message history with BOTH user and assistant turns
        model_history = build_message_history(history) if history else None

        with st.spinner("Generating response..."):
            result = await pinescript_agent.run(
                prompt,
                deps=deps,
                message_history=model_history,
                model=resolve_model(),  # OpenRouter when its key is set, like the CLI
            )
            return result.output.response, result.output.snippets_used


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Status")
setup_ok, api_key, doc_count = asyncio.run(verify_setup())

if setup_ok:
    st.sidebar.success(f"✅ Database connected with {doc_count} documents")
    masked = f"{api_key[:4]}...{api_key[-4:]}" if api_key and len(api_key) > 8 else "Not set"
    st.sidebar.success(f"✅ OpenAI API key: {masked}")
else:
    st.sidebar.warning("⚠️ Setup incomplete. Check errors above.")

if configured_key("OPENROUTER_API_KEY"):
    st.sidebar.success("✅ Using OpenRouter for queries")
else:
    st.sidebar.info("Using OpenAI for queries (no OpenRouter key found)")

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    save_chat_history([])
    st.sidebar.success("Chat history cleared!")

st.sidebar.markdown("### Example Queries")
examples = [
    "How do I create a moving average crossover strategy?",
    "How to calculate RSI in Pine Script?",
    "What's the difference between series and simple variables?",
    "How to plot markers on my chart?",
    "Explain Pine Script arrays and matrices",
]

for i, example in enumerate(examples):
    if st.sidebar.button(f"{i + 1}. {example}"[:40] + "..."):
        st.session_state.messages.append({"role": "user", "content": example})
        save_chat_history(st.session_state.messages)
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("PineScript Expert")
st.markdown("Ask questions about Pine Script programming for TradingView")

# Render existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "snippets" in message:
            st.caption(f"Used {message['snippets']} document references")

# Handle new user input
if prompt := st.chat_input("Ask about Pine Script"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            # Pass ALL previous messages (user + assistant) for context
            previous = st.session_state.messages[:-1]
            response, snippets = asyncio.run(process_query(prompt, previous))

            placeholder.markdown(response)
            st.caption(f"Used {snippets} document references")

            st.session_state.messages.append(
                {"role": "assistant", "content": response, "snippets": snippets}
            )
            save_chat_history(st.session_state.messages)

        except Exception as e:
            placeholder.markdown(f"Error: {e}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {e}", "snippets": 0}
            )
            save_chat_history(st.session_state.messages)
