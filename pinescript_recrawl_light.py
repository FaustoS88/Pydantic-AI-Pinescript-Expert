"""Lightweight PineScript docs re-crawl — no Chromium, just httpx + html2text.

Replaces Crawl4AI with plain HTTP requests. TradingView docs are server-rendered
so no JS execution is needed. Reuses the same chunking pipeline from rag_utils.

Usage:
    # Standard re-crawl (code-aware split, no LLM prefix):
    python pinescript_recrawl_light.py [--clear]

    # Contextual re-crawl (code-aware split + LLM prefix per chunk):
    python pinescript_recrawl_light.py --contextual [--clear]
    # Cost: ~$5-10, Duration: ~20-30 min (1 LLM call per chunk via OpenRouter)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import asyncpg
import html2text
import httpx
import pydantic_core
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

# Import from existing project modules
from db_schema import create_schema, run_migration  # noqa: E402
from agent import database_connect  # noqa: E402
from rag_utils import (  # noqa: E402
    code_aware_split,
    detect_content_type,
    generate_contextual_prefix,
    prepend_chunk_header,
)
from config import (  # noqa: E402
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CONTEXTUAL_MODEL,
    EMBEDDING_MODEL,
    MAX_PAGE_CONTEXT_CHARS,
    OPENROUTER_BASE_URL,
)

# --contextual flag: generate LLM context prefix per chunk (opt-in, costs ~$5-10)
_CONTEXTUAL_MODE: bool = "--contextual" in sys.argv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recrawl")

BASE_URL = "https://www.tradingview.com/pine-script-docs"

# html2text converter
H2T = html2text.HTML2Text()
H2T.ignore_links = False
H2T.ignore_images = True
H2T.body_width = 0  # no wrapping
H2T.unicode_snob = True


async def fetch_doc_urls(client: httpx.AsyncClient) -> list[str]:
    """Fetch all doc page URLs from the welcome page navigation."""
    welcome_url = f"{BASE_URL}/welcome/"
    resp = await client.get(welcome_url, follow_redirects=True, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls: set[str] = set()

    # Find nav/sidebar links
    for nav in soup.find_all(["nav", "div"], class_=["toc", "sidebar"]):
        for a in nav.find_all("a", href=True):
            href = a["href"].split("#")[0].split("?")[0]
            if not href:
                continue
            if href.startswith("/pine-script-docs"):
                full = f"https://www.tradingview.com{href}"
            elif href.startswith(BASE_URL):
                full = href
            else:
                continue
            urls.add(full.rstrip("/") + "/")

    # Also scan all links in the page body matching the docs pattern
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0].split("?")[0]
        if href.startswith("/pine-script-docs/") and "/welcome" not in href:
            full = f"https://www.tradingview.com{href}"
            urls.add(full.rstrip("/") + "/")

    return sorted(urls)


async def fetch_page_markdown(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    """Fetch a page and convert to markdown. Returns (title, markdown)."""
    resp = await client.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else url.rstrip("/").split("/")[-1].capitalize()

    # Get main content area
    main = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
    if not main:
        main = soup.body

    # Remove nav elements
    for nav in main.find_all(["nav", "header", "footer"]):
        nav.decompose()

    markdown = H2T.handle(str(main))
    return title, markdown


def split_into_sections(
    markdown: str, url: str, page_title: str
) -> list[dict]:
    """Split markdown into chunks with headers, using code-aware chunking.

    Returns dicts with keys: url, title, content, chunk_index, content_type.
    The caller may later enrich `content` with a contextual prefix and set
    `contextual_prefix` before inserting into the database.
    """
    sections: list[dict] = []
    lines = markdown.split("\n")

    # Remove page title line if present
    if lines and lines[0].startswith("# "):
        lines = lines[1:]

    # First pass: heading-based sections
    raw_sections: list[dict[str, str]] = []
    current_section: str | None = None
    current_content: list[str] = []

    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if current_section and current_content:
                content = "\n".join(current_content).strip()
                if content:
                    raw_sections.append({"title": current_section, "content": content})
            current_section = line.lstrip("#").strip()
            current_content = []
        elif current_section:
            current_content.append(line)
        else:
            current_content.append(line)

    # Save last section
    if current_section and current_content:
        content = "\n".join(current_content).strip()
        if content:
            raw_sections.append({"title": current_section, "content": content})

    # Pre-heading content
    if not current_section and current_content:
        content = "\n".join(current_content).strip()
        if content:
            raw_sections.append({"title": page_title, "content": content})

    # Whole doc fallback
    if not raw_sections and lines:
        content = "\n".join(lines).strip()
        if content:
            raw_sections.append({"title": page_title, "content": content})

    # Second pass: code-aware split + contextual headers + content type detection
    chunk_idx = 0
    for raw in raw_sections:
        section_title = raw["title"]
        # code_aware_split replaces recursive_character_split: preserves code blocks
        chunks = code_aware_split(
            raw["content"], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        for chunk in chunks:
            if not chunk.strip():
                continue
            enriched = prepend_chunk_header(page_title, section_title, chunk)
            section_id = re.sub(r"[^a-z0-9]+", "-", section_title.lower())
            sections.append(
                {
                    "url": f"{url}#{section_id}-{chunk_idx}",
                    "title": f"{page_title} - {section_title}",
                    "content": enriched,
                    "raw_chunk": chunk,  # stored for contextual prefix generation
                    "chunk_index": chunk_idx,
                    "content_type": detect_content_type(chunk),
                    "contextual_prefix": None,  # filled in process_and_store if --contextual
                }
            )
            chunk_idx += 1

    return sections


async def generate_embedding(
    openai_client: AsyncOpenAI, text: str, sem: asyncio.Semaphore
) -> list[float] | None:
    """Generate embedding with concurrency limit."""
    async with sem:
        try:
            resp = await openai_client.embeddings.create(
                input=text, model=EMBEDDING_MODEL
            )
            return resp.data[0].embedding
        except Exception as exc:
            logger.warning("Embedding error: %s", exc)
            return None


async def process_and_store(
    pool: asyncpg.Pool,
    openai_client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    url: str,
    title: str,
    markdown: str,
    contextual_client: AsyncOpenAI | None = None,
) -> int:
    """Process a page: split → (optional contextual prefix) → embed → upsert.

    Returns chunk count stored.
    """
    sections = split_into_sections(markdown, url, title)
    if not sections:
        return 0

    inserted = 0
    for section in sections:
        raw_chunk = section.get("raw_chunk", section["content"])
        embed_content = section["content"]  # default: header + chunk
        contextual_prefix: str | None = None

        # Tier 2: LLM contextual prefix (opt-in via --contextual)
        if _CONTEXTUAL_MODE and contextual_client is not None:
            async with sem:
                try:
                    prefix = await generate_contextual_prefix(
                        chunk=raw_chunk,
                        page_title=title,
                        full_page_content=markdown,
                        openai_client=contextual_client,
                        model=CONTEXTUAL_MODEL,
                    )
                    contextual_prefix = prefix
                    # Enriched content used for embedding: context marker + chunk
                    embed_content = f"[Context: {prefix}]\n\n{raw_chunk}"
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Contextual prefix failed for %s: %s", url, exc)
                    # Fall back to header-enriched content
                    embed_content = section["content"]

        embedding_text = f"title: {section['title']}\n\ncontent: {embed_content}"
        embedding = await generate_embedding(openai_client, embedding_text, sem)
        if not embedding:
            continue

        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            """
            INSERT INTO pinescript_docs
              (url, title, content, embedding, search_vector,
               chunk_index, content_type, contextual_prefix)
            VALUES ($1, $2, $3, $4,
                    to_tsvector('english', coalesce($2, '') || ' ' || coalesce($3, '')),
                    $5, $6, $7)
            ON CONFLICT (url) DO UPDATE SET
              content           = EXCLUDED.content,
              embedding         = EXCLUDED.embedding,
              search_vector     = EXCLUDED.search_vector,
              chunk_index       = EXCLUDED.chunk_index,
              content_type      = EXCLUDED.content_type,
              contextual_prefix = EXCLUDED.contextual_prefix
            """,
            section["url"],
            section["title"],
            section["content"],
            embedding_json,
            section["chunk_index"],
            section["content_type"],
            contextual_prefix,
        )
        inserted += 1

    return inserted


async def clear_database() -> None:
    """Clear pinescript_docs table."""
    logger.info("Clearing existing database entries...")
    async with database_connect(False) as pool:
        count = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
        logger.info("Found %d existing records", count)
        await pool.execute("DELETE FROM pinescript_docs")
        logger.info("Deleted %d records", count)


async def run_recrawl() -> None:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    # Embedding client — always uses OpenAI directly
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    sem = asyncio.Semaphore(5)

    # Contextual prefix client — uses OpenRouter for cheaper LLM access
    contextual_client: AsyncOpenAI | None = None
    if _CONTEXTUAL_MODE:
        openrouter_key = os.getenv("OPENROUTER_API_KEY") or openai_api_key
        contextual_client = AsyncOpenAI(
            api_key=openrouter_key,
            base_url=OPENROUTER_BASE_URL,
        )
        logger.info(
            "Contextual mode ENABLED — model: %s, max_page_chars: %d",
            CONTEXTUAL_MODEL,
            MAX_PAGE_CONTEXT_CHARS,
        )

    async with httpx.AsyncClient(
        headers={"User-Agent": "PineScriptDocsCrawler/1.0"},
        timeout=30,
    ) as http:
        # Step 1: collect URLs
        logger.info("Collecting documentation URLs...")
        urls = await fetch_doc_urls(http)
        logger.info("Found %d documentation pages", len(urls))

        if not urls:
            logger.error("No URLs found!")
            return

        # Step 2: crawl + process + store
        async with database_connect(True) as pool:
            await create_schema(pool)
            await run_migration(pool)

            total_chunks = 0
            for i, url in enumerate(urls, 1):
                try:
                    logger.info("[%d/%d] Crawling: %s", i, len(urls), url)
                    title, markdown = await fetch_page_markdown(http, url)

                    if not markdown or len(markdown) < 50:
                        logger.warning("  → Empty or too short, skipping")
                        continue

                    chunks = await process_and_store(
                        pool, openai_client, sem, url, title, markdown,
                        contextual_client=contextual_client,
                    )
                    total_chunks += chunks
                    logger.info("  → %d chunks stored (total: %d)", chunks, total_chunks)

                    # Rate limit: 1 second between pages
                    await asyncio.sleep(1)

                except Exception as exc:
                    logger.error("  → Error: %s", exc)

            final_count = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
            logger.info("Re-crawl complete — %d total chunks in database", final_count)


async def main() -> None:
    if "--clear" in sys.argv:
        await clear_database()
    await run_recrawl()


if __name__ == "__main__":
    asyncio.run(main())
