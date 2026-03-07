"""Lightweight PineScript docs re-crawl — no Chromium, just httpx + html2text.

Replaces Crawl4AI with plain HTTP requests. TradingView docs are server-rendered
so no JS execution is needed. Reuses the same chunking pipeline from rag_utils.

Usage:
    python pinescript_recrawl_light.py [--clear]
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from typing import Any

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
from rag_utils import prepend_chunk_header, recursive_character_split  # noqa: E402
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL  # noqa: E402

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
) -> list[dict[str, str]]:
    """Split markdown into chunks with headers — same logic as original crawler."""
    sections: list[dict[str, str]] = []
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

    # Second pass: recursive split + contextual headers
    chunk_idx = 0
    for raw in raw_sections:
        section_title = raw["title"]
        chunks = recursive_character_split(
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
) -> int:
    """Process a page: split → embed → insert. Returns chunk count."""
    sections = split_into_sections(markdown, url, title)
    if not sections:
        return 0

    inserted = 0
    for section in sections:
        embedding_text = f"title: {section['title']}\n\ncontent: {section['content']}"
        embedding = await generate_embedding(openai_client, embedding_text, sem)
        if not embedding:
            continue

        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            """
            INSERT INTO pinescript_docs (url, title, content, embedding, search_vector)
            VALUES ($1, $2, $3, $4,
                    to_tsvector('english', coalesce($2, '') || ' ' || coalesce($3, '')))
            ON CONFLICT (url) DO NOTHING
            """,
            section["url"],
            section["title"],
            section["content"],
            embedding_json,
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

    openai_client = AsyncOpenAI(api_key=openai_api_key)
    sem = asyncio.Semaphore(5)

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
                        pool, openai_client, sem, url, title, markdown
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
