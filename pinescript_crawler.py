from __future__ import annotations

import asyncio
import os
import sys
import json
import re
import pydantic_core
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, AsyncGenerator, Set

import asyncpg
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from datetime import datetime

# Import crawler from crawl4ai
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

# Force reload environment variables
load_dotenv(override=True)

# Import database setup functions
from db_schema import create_schema
from agent import database_connect

class PineScriptDocsCrawler:
    """Pine Script Documentation Crawler with processing capabilities"""
    
    def __init__(self):
        # This is the correct base URL that works in the original script
        self.base_url = "https://www.tradingview.com/pine-script-docs"
        self.output_dir = "pinescript_docs"
        self.visited_urls: Set[str] = set()
        
        # Initialize OpenAI client for embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("WARNING: OPENAI_API_KEY environment variable is not found!")
            print("Please enter your OpenAI API key:")
            openai_api_key = input("> ")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is required to continue")
            # Save to environment
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize client
        self.openai = AsyncOpenAI(api_key=openai_api_key)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define the extraction schema for structure
        self.structure_schema = {
            "name": "PineScript Documentation",
            "baseSelector": "main",  # Main content area
            "fields": [
                {
                    "name": "title",
                    "selector": "h1",
                    "type": "text"
                },
                {
                    "name": "content",
                    "selector": "main > div",  # Main content excluding navigation
                    "type": "html"
                },
                {
                    "name": "navigation",
                    "selector": "nav",  # Left navigation menu
                    "type": "html"
                },
                {
                    "name": "toc",
                    "selector": "[aria-label='Table of contents']",  # Right-side TOC
                    "type": "html"
                }
            ]
        }
        
        # Semaphore to limit concurrent embedding API calls
        self.sem = asyncio.Semaphore(5)
    
    def normalize_url(self, url: str) -> str:
        """Convert relative URLs to absolute and clean them"""
        if not url:
            return ""
            
        # Remove anchor tags and query parameters
        url = url.split('#')[0].split('?')[0]
        
        # Skip external links and special protocols
        if url.startswith(('http', 'https')) and not url.startswith(self.base_url):
            return ""
        if url.startswith(('mailto:', 'tel:', 'javascript:')):
            return ""
            
        # Handle relative URLs
        if not url.startswith('http'):
            if url.startswith('/'):
                url = f"https://www.tradingview.com{url}"
            else:
                url = f"{self.base_url}/{url}"
                
        return url
    
    async def get_all_doc_urls(self) -> List[str]:
        """Extract all documentation URLs from the navigation menu"""
        urls = set()
        print("Starting to collect URLs...")
        
        # Start with main sections from left navigation
        browser_config = BrowserConfig(
            headless=True,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
        )
        
        # THIS IS CRITICAL - the /welcome/ path is the entry point that works
        welcome_url = f"{self.base_url}/welcome/"
        print(f"Starting from: {welcome_url}")
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=welcome_url)
            if result.success:
                print("Successfully accessed the main page")
                soup = BeautifulSoup(result.html, 'html.parser')
                
                # Find all navigation elements
                nav_elements = soup.find_all(['nav', 'div'], class_=['toc', 'sidebar'])
                for nav in nav_elements:
                    for link in nav.find_all('a'):
                        href = link.get('href')
                        if href:
                            full_url = self.normalize_url(href)
                            if full_url:
                                urls.add(full_url)
                                print(f"Found URL: {full_url}")
            else:
                print(f"Failed to access main page: {result.error_message}")
                
                # Fallback to a v5 link if v6 doesn't work
                fallback_url = f"{self.base_url}/en/v5/welcome/"
                print(f"Trying fallback: {fallback_url}")
                
                fallback_result = await crawler.arun(url=fallback_url)
                if fallback_result.success:
                    print("Successfully accessed fallback page")
                    soup = BeautifulSoup(fallback_result.html, 'html.parser')
                    
                    # Find all navigation elements
                    nav_elements = soup.find_all(['nav', 'div'], class_=['toc', 'sidebar'])
                    for nav in nav_elements:
                        for link in nav.find_all('a'):
                            href = link.get('href')
                            if href:
                                full_url = self.normalize_url(href)
                                if full_url:
                                    urls.add(full_url)
                                    print(f"Found URL: {full_url}")
        
        # If we found no URLs, try some common paths
        if not urls:
            print("No URLs found in navigation, trying common paths...")
            common_paths = [
                "welcome/", "introduction/", "concepts/", "language/", 
                "essential/", "resources/", "reference/", "faq/"
            ]
            
            for path in common_paths:
                urls.add(f"{self.base_url}/{path}")
        
        urls_list = sorted(list(urls))
        print(f"Total URLs found: {len(urls_list)}")
        return urls_list
    
    # NEW PROCESSING METHODS
    
    def clean_navigation(self, text):
        """Remove navigation elements and links"""
        # Remove navigation sections
        text = re.sub(r'Version Version.*?Auto', '', text, flags=re.DOTALL)
        text = re.sub(r'\* \[.*?\n', '', text)
        text = re.sub(r'Copyright © .*?TradingView.*?\n', '', text)
        text = re.sub(r'On this page.*?\n', '', text)
        # Clean up additional navigation elements
        text = re.sub(r'\[ User Manual \].*?\n', '', text)
        text = re.sub(r'\[ Previous .*? \]', '', text)
        text = re.sub(r'\[ Next .*? \]', '', text)
        return text
        
    def extract_code_blocks(self, text):
        """Preserve and clean code blocks"""
        # Find Pine Script code blocks
        code_blocks = re.findall(r'```(?:pine)?(.*?)```', text, re.DOTALL)
        clean_blocks = []
        for block in code_blocks:
            # Clean the code block
            clean_block = block.strip()
            if clean_block:
                clean_blocks.append(f"```pine\n{clean_block}\n```")
        return clean_blocks
        
    def extract_function_docs(self, text):
        """Extract function documentation"""
        # Find function descriptions
        functions = re.findall(r'@function.*?@returns.*?\n', text, re.DOTALL)
        return functions
        
    def process_content(self, raw_content, url):
        """Process raw content into a cleaner format for embedding"""
        # Skip if no real content
        if len(raw_content) < 100 or 'User Manual' not in raw_content:
            return raw_content
            
        # Clean navigation and basic structure
        content = self.clean_navigation(raw_content)
        
        # Extract valuable parts
        code_blocks = self.extract_code_blocks(content)
        function_docs = self.extract_function_docs(content)
        
        # Extract main content sections (Q&A format in FAQ)
        sections = re.findall(r'##\s+\[(.*?)\].*?\n(.*?)(?=##|\Z)', content, re.DOTALL)
        
        # Build processed content
        processed = []
        
        if sections:
            for title, section in sections:
                if any(keyword in section.lower() for keyword in ['pine', 'script', 'function', 'indicator', 'value', 'parameter']):
                    clean_section = re.sub(r'\[\^.*?\]', '', section)  # Remove footnotes
                    clean_section = re.sub(r'\(https://.*?\)', '', clean_section)  # Remove links
                    processed.append(f"## {title}\n{clean_section.strip()}")
        
        if code_blocks:
            processed.append("\n## Code Examples\n")
            processed.extend(code_blocks)
            
        if function_docs:
            processed.append("\n## Function Documentation\n")
            processed.extend(function_docs)
            
        # If nothing valuable extracted, return original content
        if not processed:
            return raw_content
            
        return "\n\n".join(processed)
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API"""
        async with self.sem:
            try:
                # Limit text length to avoid token limit issues
                max_length = 16000  # About 4000 tokens
                if len(text) > max_length:
                    print(f"Truncating text from {len(text)} to {max_length} characters")
                    text = text[:max_length]
                
                response = await self.openai.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error generating embedding: {e}")
                return []
    
    async def crawl_docs(self, urls: List[str]):
        """Crawl documentation pages and store in vector database"""
        print("Starting crawling process...")
        
        # Configure extraction strategy
        structure_strategy = JsonCssExtractionStrategy(
            schema=self.structure_schema,
            verbose=True
        )
        
        browser_config = BrowserConfig(
            headless=True,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = f"{self.output_dir}/all_docs_{timestamp}.md"
        failed_path = f"{self.output_dir}/failed_urls_{timestamp}.txt"
        
        # Connect to database
        async with database_connect(True) as pool:
            # Create schema if needed
            await create_schema(pool)
            
            # Process URLs
            async with AsyncWebCrawler(config=browser_config) as crawler:
                success = 0
                failed = 0
                
                with open(combined_path, "w", encoding="utf-8") as combined_file, \
                     open(failed_path, "w", encoding="utf-8") as failed_file:
                    
                    # Process in small batches
                    batch_size = 3
                    for i in range(0, len(urls), batch_size):
                        batch = urls[i:i + batch_size]
                        print(f"\nProcessing batch {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size}")
                        
                        for url in batch:
                            try:
                                if url in self.visited_urls:
                                    print(f"Already visited {url}, skipping")
                                    continue
                                    
                                self.visited_urls.add(url)
                                print(f"Crawling: {url}")
                                
                                # Use the extraction strategy
                                result = await crawler.arun(
                                    url=url,
                                    extraction_strategy=structure_strategy
                                )
                                
                                if result.success:
                                    # Get page name for file
                                    page_name = url.rstrip('/').split('/')[-1] or 'index'
                                    file_path = f"{self.output_dir}/{page_name}_{timestamp}.md"
                                    
                                    # Get the markdown content
                                    markdown = result.markdown_v2.raw_markdown
                                    
                                    # Save individual file
                                    with open(file_path, "w", encoding="utf-8") as f:
                                        f.write(f"# {page_name}\n\n")
                                        f.write(f"Source: {url}\n\n")
                                        f.write(markdown)
                                    
                                    # Add to combined file
                                    combined_file.write(f"\n\n# {page_name}\n\n")
                                    combined_file.write(f"Source: {url}\n\n")
                                    combined_file.write(markdown)
                                    combined_file.write("\n\n---\n\n")
                                    
                                    # Process for vector database
                                    await self.process_and_store_document(url, markdown, pool)
                                    
                                    success += 1
                                    print(f"Successfully processed: {page_name}")
                                else:
                                    print(f"Failed to crawl {url}: {result.error_message}")
                                    failed_file.write(f"{url}: {result.error_message}\n")
                                    failed += 1
                                    
                            except Exception as e:
                                print(f"Error processing {url}: {str(e)}")
                                failed_file.write(f"{url}: {str(e)}\n")
                                failed += 1
                        
                        # Rate limiting between batches
                        await asyncio.sleep(2)
                
                # Count documents in database
                count = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
                print(f"\nCrawling completed:")
                print(f"- Successfully processed: {success} pages")
                print(f"- Failed: {failed} pages")
                print(f"- Database now contains {count} document sections")
            
    async def process_and_store_document(self, url: str, markdown: str, pool: asyncpg.Pool):
        """Process a document and store its chunks in the database"""
        if not markdown:
            print(f"No content for {url}, skipping")
            return
        
        # Process the content before splitting and storing
        processed_content = self.process_content(markdown, url)
            
        # Split document into sections based on headings
        sections = self.split_into_sections(processed_content, url)
        
        if not sections:
            print(f"No valid sections found for {url}")
            return
            
        print(f"Processing {len(sections)} sections from {url}")
        
        # Process each section
        for section in sections:
            # Check if already exists
            exists = await pool.fetchval(
                "SELECT 1 FROM pinescript_docs WHERE url = $1",
                section["url"]
            )
            
            if exists:
                print(f"Section already exists: {section['url']}")
                continue
            
            # Generate embedding
            embedding_text = f"title: {section['title']}\n\ncontent: {section['content']}"
            embedding = await self.generate_embedding(embedding_text)
            
            if not embedding:
                print(f"Failed to generate embedding for {section['url']}")
                continue
            
            # Convert to JSON
            embedding_json = pydantic_core.to_json(embedding).decode()
            
            # Insert into database
            await pool.execute(
                """
                INSERT INTO pinescript_docs (url, title, content, embedding)
                VALUES ($1, $2, $3, $4)
                """,
                section["url"],
                section["title"],
                section["content"],
                embedding_json
            )
            
            print(f"Inserted section: {section['title']}")
    
    def split_into_sections(self, markdown: str, url: str) -> List[Dict[str, str]]:
        """Split markdown into sections based on headings"""
        sections = []
        lines = markdown.split("\n")
        
        # Extract title from first heading or URL
        page_title = url.split("/")[-1].replace(".html", "").capitalize()
        if lines and lines[0].startswith("# "):
            page_title = lines[0][2:].strip()
            lines = lines[1:]
        
        current_section = None
        current_content = []
        
        for line in lines:
            # Check for section headings
            if line.startswith("## "):
                # Save previous section
                if current_section and current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        section_id = re.sub(r'[^a-z0-9]+', '-', current_section.lower())
                        sections.append({
                            "url": f"{url}#{section_id}",
                            "title": f"{page_title} - {current_section}",
                            "content": content
                        })
                
                # Start new section
                current_section = line[3:].strip()
                current_content = []
            # Check for subsections
            elif line.startswith("### "):
                # Don't start a new main section, just add this as a subsection marker
                if current_section:
                    current_content.append(line)
            # Add content to current section
            elif current_section:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            content = "\n".join(current_content).strip()
            if content:
                section_id = re.sub(r'[^a-z0-9]+', '-', current_section.lower())
                sections.append({
                    "url": f"{url}#{section_id}",
                    "title": f"{page_title} - {current_section}",
                    "content": content
                })
        
        # If no sections found, use the entire document
        if not sections and lines:
            content = "\n".join(lines).strip()
            if content:
                sections.append({
                    "url": url,
                    "title": page_title,
                    "content": content
                })
        
        return sections
    
    async def run(self):
        """Main execution method"""
        print("Starting PineScript documentation crawler...")
        urls = await self.get_all_doc_urls()
        if not urls:
            print("No documentation pages found!")
            return
            
        print(f"\nFound {len(urls)} documentation pages")
        await self.crawl_docs(urls)

async def clear_database():
    """Clear the pinescript_docs table"""
    print("Clearing existing database entries...")
    
    async with database_connect(False) as pool:
        # Count before deletion
        count_before = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
        print(f"Found {count_before} existing records")
        
        # Delete all records
        await pool.execute("DELETE FROM pinescript_docs")
        
        # Count after deletion
        count_after = await pool.fetchval("SELECT COUNT(*) FROM pinescript_docs")
        print(f"Deleted {count_before - count_after} records")
        print(f"Database now contains {count_after} records")

async def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        # Clear the database first
        await clear_database()
    
    crawler = PineScriptDocsCrawler()
    await crawler.run()

if __name__ == "__main__":
    asyncio.run(main())
