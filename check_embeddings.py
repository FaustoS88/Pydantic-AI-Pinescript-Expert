#!/usr/bin/env python3
"""
Quick embedding dimension checker.
Run this to check if your embedding model matches your database.
"""

import asyncio
from fix_embeddings import check_embedding_dimensions

if __name__ == "__main__":
    asyncio.run(check_embedding_dimensions())