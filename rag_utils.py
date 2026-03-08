"""Advanced RAG utilities — hybrid search, cross-encoder reranking, MMR deduplication.

This module implements Tier 1 and Tier 2 RAG improvements:
- Reciprocal Rank Fusion (RRF) for hybrid BM25 + vector search
- Cross-encoder reranking with sentence-transformers
- Maximal Marginal Relevance (MMR) for result diversity
- Recursive character splitting with overlap
- Contextual chunk headers
- [Tier 2] Code-aware chunking (never splits inside fenced code blocks)
- [Tier 2] Contextual Retrieval — LLM-generated prefix per chunk (Anthropic, Sep 2024)
- [Tier 2] Content type detection (reference / example / tutorial)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTEXTUAL_MODEL,
    HYBRID_SEARCH_ALPHA,
    MAX_PAGE_CONTEXT_CHARS,
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

    Format: "Document: PineScript v6 Reference | Section: {page_title} > {section}\n\n{content}"
    """
    if section_title and section_title != page_title:
        header = f"Document: PineScript v6 Reference | Section: {page_title} > {section_title}"
    else:
        header = f"Document: PineScript v6 Reference | Section: {page_title}"
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


# ---------------------------------------------------------------------------
# Tier 2A. Code-aware chunking — never splits inside fenced code blocks
# ---------------------------------------------------------------------------


def code_aware_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text while preserving fenced code blocks intact.

    Fenced code blocks (``` ... ```) are NEVER split even if they exceed
    chunk_size.  Prose sections delegate to _split_recursive for normal
    hierarchical splitting.  Inline code (single backtick) is treated as prose.

    Algorithm:
    1. Parse text into alternating ('prose', ...) / ('code', ...) segments
       by splitting on triple-backtick boundaries.
    2. Greedily accumulate segments into a working chunk.
    3. When a prose segment would overflow: flush current chunk, split prose
       with _split_recursive, continue.
    4. When a code segment would overflow: flush current chunk, emit the code
       block as its own chunk (intact, even if oversized).

    Args:
        text: Source text (may contain fenced code blocks).
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Characters of overlap to add between adjacent chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    # Split on ``` boundaries; even-index parts are prose, odd-index are code
    parts = text.split("```")
    segments: list[tuple[str, str]] = []
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2 == 0:
            segments.append(("prose", part))
        else:
            # Reconstruct the fenced block with its language tag
            segments.append(("code", "```" + part + "```"))

    if not segments:
        return [text]

    chunks: list[str] = []
    current = ""

    for seg_type, seg_content in segments:
        if seg_type == "code":
            candidate = current + seg_content
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                # Flush current prose first, then emit code block whole
                if current.strip():
                    chunks.append(current.strip())
                    current = ""
                chunks.append(seg_content.strip())
        else:
            # Prose segment
            candidate = current + seg_content
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                # Flush current chunk
                if current.strip():
                    chunks.append(current.strip())
                    current = ""
                # Recursively split the prose segment
                sub = _split_recursive(seg_content.strip(), _SEPARATORS, chunk_size, chunk_overlap)
                if sub:
                    # Keep the last sub-chunk as the new current to allow
                    # subsequent content to be appended (maintain continuity)
                    chunks.extend(sub[:-1])
                    current = sub[-1]

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Tier 2B. Contextual Retrieval — LLM-generated prefix per chunk
# ---------------------------------------------------------------------------


async def generate_contextual_prefix(
    chunk: str,
    page_title: str,
    full_page_content: str,
    openai_client,
    model: str = CONTEXTUAL_MODEL,
) -> str:
    """Generate a one-sentence LLM context description for a chunk.

    Implements Anthropic's Contextual Retrieval approach (Sep 2024):
    -49% retrieval failures standalone, -67% combined with reranking.

    The returned string is the raw prefix sentence (NOT the enriched content).
    The caller is responsible for:
      1. Prepending "[Context: {prefix}]\\n\\n" to the chunk for embedding.
      2. Storing the prefix separately in the `contextual_prefix` DB column.

    On LLM failure: returns the page_title as a minimal fallback prefix so
    indexing never crashes.

    Args:
        chunk: The chunk text to contextualise (first 600 chars used in prompt).
        page_title: Title of the source page.
        full_page_content: Full page markdown (truncated to MAX_PAGE_CONTEXT_CHARS).
        openai_client: AsyncOpenAI-compatible client (any OpenAI-compatible API).
        model: LLM model to use for prefix generation.

    Returns:
        A concise 1-sentence description of the chunk (30-80 words).
    """
    page_excerpt = full_page_content[:MAX_PAGE_CONTEXT_CHARS]
    prompt = (
        "Here is a page from the PineScript v6 documentation.\n"
        f"Page title: {page_title}\n"
        f"Page content (excerpt):\n{page_excerpt}\n\n"
        f"Here is a specific chunk from this page:\n{chunk[:600]}\n\n"
        "Write one concise sentence (30-80 words) describing what this chunk covers. "
        "Be specific: name the function, concept, or language feature. "
        'Start with "PineScript" or the function name. Reply with ONLY the sentence.'
    )

    try:
        resp = await openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.1,
        )
        prefix = resp.choices[0].message.content.strip()
        return prefix if prefix else page_title
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_contextual_prefix: LLM call failed, using fallback: %s", exc)
        return page_title


# ---------------------------------------------------------------------------
# Tier 2C. Content type detection
# ---------------------------------------------------------------------------


def detect_content_type(content: str) -> str:
    """Classify chunk content as 'reference', 'example', or 'tutorial'.

    Heuristic rules based on code block density and keyword presence:
    - 'example'   : 2+ code blocks AND short (<200 words)  — code-heavy snippet
    - 'reference' : param/syntax keywords AND at least 1 code block — API ref
    - 'tutorial'  : everything else — explanatory prose

    Args:
        content: The chunk text.

    Returns:
        One of: 'reference', 'example', 'tutorial'.
    """
    if not content:
        return "tutorial"

    code_blocks = content.count("```") // 2
    has_param_keywords = any(
        k in content.lower()
        for k in ["parameter", "argument", "syntax", "returns", "return type"]
    )
    word_count = len(content.split())

    if code_blocks >= 2 and word_count < 200:
        return "example"
    if has_param_keywords and code_blocks >= 1:
        return "reference"
    return "tutorial"
