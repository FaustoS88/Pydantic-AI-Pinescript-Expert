"""Advanced RAG utilities — hybrid search, cross-encoder reranking, MMR deduplication.

This module implements Tier 1 RAG improvements:
- Reciprocal Rank Fusion (RRF) for hybrid BM25 + vector search
- Cross-encoder reranking with sentence-transformers
- Maximal Marginal Relevance (MMR) for result diversity
- Recursive character splitting with overlap
- Contextual chunk headers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HYBRID_SEARCH_ALPHA,
    MMR_LAMBDA,
    RERANK_MODEL,
    RERANK_TOP_N,
    RETRIEVAL_CANDIDATES,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievedDoc:
    """A document returned from the retrieval pipeline."""

    url: str
    title: str
    content: str
    vector_score: float = 0.0  # cosine similarity (0–1)
    bm25_score: float = 0.0  # ts_rank score
    rrf_score: float = 0.0  # fused score
    rerank_score: float = 0.0  # cross-encoder score
    embedding: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Contextual Chunk Headers
# ---------------------------------------------------------------------------


def prepend_chunk_header(page_title: str, section_title: str, content: str) -> str:
    """Prepend document context to a chunk for better embedding quality.

    Format: "Document: PineScript v5 Reference | Section: {page_title} > {section}\n\n{content}"
    """
    if section_title and section_title != page_title:
        header = f"Document: PineScript v5 Reference | Section: {page_title} > {section_title}"
    else:
        header = f"Document: PineScript v5 Reference | Section: {page_title}"
    return f"{header}\n\n{content}"


# ---------------------------------------------------------------------------
# 2. Hybrid Search — Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

RRF_K = 60  # standard constant for RRF


def reciprocal_rank_fusion(
    docs: list[RetrievedDoc],
    alpha: float = HYBRID_SEARCH_ALPHA,
) -> list[RetrievedDoc]:
    """Fuse vector and BM25 rankings using Reciprocal Rank Fusion.

    Each doc gets:  score = alpha * (1 / (k + vector_rank)) + (1 - alpha) * (1 / (k + bm25_rank))

    Args:
        docs: Documents with vector_score and bm25_score already set.
        alpha: Weight for vector vs BM25 (0.5 = equal).

    Returns:
        Documents sorted by fused RRF score (descending).
    """
    if not docs:
        return []

    # Rank by vector score (descending) — highest similarity = rank 1
    by_vector = sorted(docs, key=lambda d: d.vector_score, reverse=True)
    for rank, doc in enumerate(by_vector, start=1):
        doc.rrf_score = alpha * (1.0 / (RRF_K + rank))

    # Rank by BM25 score (descending)
    by_bm25 = sorted(docs, key=lambda d: d.bm25_score, reverse=True)
    for rank, doc in enumerate(by_bm25, start=1):
        doc.rrf_score += (1.0 - alpha) * (1.0 / (RRF_K + rank))

    return sorted(docs, key=lambda d: d.rrf_score, reverse=True)


# ---------------------------------------------------------------------------
# 3. Similarity Threshold
# ---------------------------------------------------------------------------


def apply_similarity_threshold(
    docs: list[RetrievedDoc],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[RetrievedDoc]:
    """Filter out documents below the cosine similarity threshold."""
    filtered = [d for d in docs if d.vector_score >= threshold]
    if len(filtered) < len(docs):
        logger.debug(
            "Similarity threshold %.2f filtered %d → %d docs",
            threshold,
            len(docs),
            len(filtered),
        )
    return filtered


# ---------------------------------------------------------------------------
# 4. Cross-Encoder Reranking
# ---------------------------------------------------------------------------

_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load the cross-encoder model (heavy import)."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder model: %s", RERANK_MODEL)
            _cross_encoder = CrossEncoder(RERANK_MODEL)
            logger.info("Cross-encoder loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
            _cross_encoder = False  # sentinel: tried and failed
    return _cross_encoder if _cross_encoder is not False else None


def rerank_docs(
    query: str,
    docs: list[RetrievedDoc],
    top_n: int = RERANK_TOP_N,
) -> list[RetrievedDoc]:
    """Rerank documents using a cross-encoder model.

    Args:
        query: The user's search query.
        docs: Candidate documents from retrieval.
        top_n: Number of results to return after reranking.

    Returns:
        Top-n documents sorted by cross-encoder relevance score.
    """
    if not docs:
        return []

    encoder = _get_cross_encoder()
    if encoder is None:
        logger.debug("Cross-encoder unavailable, returning docs as-is")
        return docs[:top_n]

    pairs = [[query, doc.content] for doc in docs]
    scores = encoder.predict(pairs)

    for doc, score in zip(docs, scores):
        doc.rerank_score = float(score)

    reranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
    return reranked[:top_n]


# ---------------------------------------------------------------------------
# 5. Recursive Character Splitting with Overlap
# ---------------------------------------------------------------------------

# Separators tried in order: double newline, single newline, sentence end, space
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def recursive_character_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text recursively using hierarchical separators with overlap.

    Tries to split on paragraph breaks first, then newlines, then sentences,
    then spaces. Each chunk gets `chunk_overlap` characters of overlap with
    the previous chunk.

    Args:
        text: The text to split.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Base case: text fits in one chunk
    if len(text) <= chunk_size:
        return [text]

    return _split_recursive(text, _SEPARATORS, chunk_size, chunk_overlap)


def _split_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Internal recursive splitting logic."""
    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()]

    # Find the best separator that exists in the text
    chosen_sep = None
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    # If no separator found, hard-split by chunk_size
    if chosen_sep is None:
        return _hard_split(text, chunk_size, chunk_overlap)

    # Split by the chosen separator
    parts = text.split(chosen_sep)
    remaining_separators = separators[separators.index(chosen_sep) + 1 :]

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{chosen_sep}{part}" if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Save current chunk if non-empty
            if current.strip():
                chunks.append(current.strip())

            # If this single part is too large, recurse with finer separators
            if len(part) > chunk_size and remaining_separators:
                sub_chunks = _split_recursive(
                    part, remaining_separators, chunk_size, chunk_overlap
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    # Don't forget the last chunk
    if current.strip():
        chunks.append(current.strip())

    # Apply overlap between adjacent chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, chunk_overlap)

    return chunks


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Hard-split text when no separator is found."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap if chunk_overlap > 0 else end
        if start >= end:
            break
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap from the end of each chunk to the start of the next."""
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:] if len(chunks[i - 1]) >= overlap else chunks[i - 1]
        # Only prepend overlap if it doesn't duplicate the start
        if not chunks[i].startswith(prev_tail):
            result.append(prev_tail + " " + chunks[i])
        else:
            result.append(chunks[i])
    return result


# ---------------------------------------------------------------------------
# 6. Maximal Marginal Relevance (MMR)
# ---------------------------------------------------------------------------


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def mmr_select(
    query_embedding: list[float],
    docs: list[RetrievedDoc],
    top_n: int = RERANK_TOP_N,
    lambda_param: float = MMR_LAMBDA,
) -> list[RetrievedDoc]:
    """Select documents using Maximal Marginal Relevance for diversity.

    MMR score = lambda * sim(query, doc) - (1 - lambda) * max(sim(doc, selected))

    Args:
        query_embedding: The query embedding vector.
        docs: Candidate documents with embeddings.
        top_n: Number of documents to select.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).

    Returns:
        Selected documents maximizing both relevance and diversity.
    """
    if not docs or not query_embedding:
        return docs[:top_n] if docs else []

    # Filter docs that have embeddings
    docs_with_emb = [d for d in docs if d.embedding]
    docs_without_emb = [d for d in docs if not d.embedding]

    if not docs_with_emb:
        return docs[:top_n]

    # Compute relevance scores (query ↔ doc)
    relevance = [_cosine_sim(query_embedding, d.embedding) for d in docs_with_emb]

    selected: list[int] = []
    remaining = list(range(len(docs_with_emb)))

    for _ in range(min(top_n, len(docs_with_emb))):
        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            rel_score = relevance[idx]

            # Max similarity to already-selected documents
            if selected:
                max_sim = max(
                    _cosine_sim(docs_with_emb[idx].embedding, docs_with_emb[s].embedding)
                    for s in selected
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)

    result = [docs_with_emb[i] for i in selected]

    # Append docs without embeddings at the end (up to top_n)
    remaining_slots = top_n - len(result)
    if remaining_slots > 0 and docs_without_emb:
        result.extend(docs_without_emb[:remaining_slots])

    return result


# ---------------------------------------------------------------------------
# Full retrieval pipeline
# ---------------------------------------------------------------------------


async def hybrid_retrieve(
    pool,
    openai_client,
    query: str,
    embedding_model: str = "text-embedding-3-small",
    candidates: int = RETRIEVAL_CANDIDATES,
    alpha: float = HYBRID_SEARCH_ALPHA,
    threshold: float = SIMILARITY_THRESHOLD,
    top_n: int = RERANK_TOP_N,
    mmr_lambda: float = MMR_LAMBDA,
    enable_rerank: bool = True,
    enable_mmr: bool = True,
) -> list[RetrievedDoc]:
    """Full hybrid retrieval pipeline: vector + BM25 → RRF → threshold → rerank → MMR.

    Args:
        pool: asyncpg connection pool.
        openai_client: AsyncOpenAI client for embeddings.
        query: User's search query.
        embedding_model: OpenAI embedding model name.
        candidates: Number of initial candidates to retrieve.
        alpha: Hybrid search weight (1.0 = vector only, 0.0 = BM25 only).
        threshold: Minimum cosine similarity to keep a result.
        top_n: Final number of results to return.
        mmr_lambda: MMR diversity parameter.
        enable_rerank: Whether to use cross-encoder reranking.
        enable_mmr: Whether to use MMR deduplication.

    Returns:
        List of RetrievedDoc objects, ordered by relevance.
    """
    import pydantic_core

    # 1. Generate query embedding
    embedding_resp = await openai_client.embeddings.create(
        input=query, model=embedding_model
    )
    query_embedding = embedding_resp.data[0].embedding
    embedding_json = pydantic_core.to_json(query_embedding).decode()

    # 2. Hybrid query: vector cosine + BM25 tsvector
    sql = """
        SELECT url, title, content, embedding,
               1 - (embedding <=> $1::vector) AS vector_score,
               ts_rank(search_vector, plainto_tsquery('english', $2)) AS bm25_score
        FROM pinescript_docs
        WHERE 1 - (embedding <=> $1::vector) > $3
        ORDER BY 1 - (embedding <=> $1::vector) DESC
        LIMIT $4
    """
    rows = await pool.fetch(sql, embedding_json, query, threshold, candidates)

    if not rows:
        return []

    # 3. Build RetrievedDoc objects
    docs = []
    for row in rows:
        emb = row["embedding"] if row["embedding"] else []
        # asyncpg may return embedding as string — parse if needed
        if isinstance(emb, str):
            emb = [float(x) for x in emb.strip("[]").split(",") if x.strip()]
        docs.append(
            RetrievedDoc(
                url=row["url"],
                title=row["title"],
                content=row["content"],
                vector_score=float(row["vector_score"]),
                bm25_score=float(row["bm25_score"]),
                embedding=emb,
            )
        )

    # 4. Reciprocal Rank Fusion
    docs = reciprocal_rank_fusion(docs, alpha=alpha)

    # 5. Cross-encoder reranking
    if enable_rerank:
        docs = rerank_docs(query, docs, top_n=min(top_n * 2, len(docs)))

    # 6. MMR deduplication
    if enable_mmr and query_embedding:
        docs = mmr_select(query_embedding, docs, top_n=top_n, lambda_param=mmr_lambda)
    else:
        docs = docs[:top_n]

    return docs
