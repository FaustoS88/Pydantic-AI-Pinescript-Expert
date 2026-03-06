# RAG Improvements — Tier 1

Six high-impact, low-effort improvements to the PineScript Expert retrieval pipeline.

## Overview

| # | Improvement | Expected Impact | Status |
|---|-------------|-----------------|--------|
| 1 | Contextual Chunk Headers | +28% retrieval quality | Implemented |
| 2 | Hybrid Search (BM25 + Vector) | +20-30% accuracy | Implemented |
| 3 | Similarity Threshold | Eliminates irrelevant noise | Implemented |
| 4 | Cross-Encoder Reranking | Major precision boost | Implemented |
| 5 | Recursive Chunking + Overlap | Fixes truncation/context loss | Implemented |
| 6 | MMR Deduplication | Eliminates redundant results | Implemented |

## 1. Contextual Chunk Headers

**File:** `rag_utils.py` — `prepend_chunk_header()`
**Modified:** `pinescript_crawler.py` — `split_into_sections()`

Each chunk gets a contextual header prepended **before embedding**, so the embedding captures the document context:

```
Document: PineScript v5 Reference | Section: Technical Analysis > ta.ema()

The ta.ema() function returns the exponentially weighted moving average...
```

This eliminates the "orphan chunk" problem where embeddings don't know which page/section the content came from.

## 2. Hybrid Search (BM25 + Vector)

**File:** `rag_utils.py` — `reciprocal_rank_fusion()`
**Schema:** `db_schema.py` — added `search_vector tsvector` column + GIN index
**Query:** `rag_utils.py` — `hybrid_retrieve()` uses both `<=>` (cosine) and `ts_rank()`

PostgreSQL's built-in `tsvector` + `ts_rank()` provides BM25-equivalent full-text search at zero infrastructure cost. Results are fused using Reciprocal Rank Fusion (RRF):

```
score = alpha / (k + vector_rank) + (1 - alpha) / (k + bm25_rank)
```

Default `alpha=0.5` (equal weight). Configurable via `PINESCRIPT_HYBRID_ALPHA` env var.

**Why critical:** `ta.rsi()` needs exact keyword match; "relative strength" needs semantic match. Hybrid search handles both.

## 3. Similarity Threshold

**File:** `rag_utils.py` — `apply_similarity_threshold()`
**Query:** SQL `WHERE 1 - (embedding <=> $1) > 0.3`

Previously returned 8 results regardless of relevance. Now discards anything below cosine similarity 0.3. Also fixed the index from L2 (`vector_l2_ops`) to cosine (`vector_cosine_ops`).

## 4. Cross-Encoder Reranking

**File:** `rag_utils.py` — `rerank_docs()`
**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, runs on CPU)

Two-stage retrieval: vector search returns 15 candidates, cross-encoder scores each `(query, document)` pair, returns top 5. The cross-encoder model is lazy-loaded on first use (~57ms inference).

**Graceful degradation:** If `sentence-transformers` is not installed, reranking is skipped and results are returned as-is.

## 5. Recursive Chunking with Overlap

**File:** `rag_utils.py` — `recursive_character_split()`
**Modified:** `pinescript_crawler.py` — `split_into_sections()`

Replaced h2-only splitting with recursive character splitting:
- **Separators (in order):** `\n\n` → `\n` → `. ` → ` `
- **chunk_size:** 500 characters (configurable via `PINESCRIPT_CHUNK_SIZE`)
- **chunk_overlap:** 50 characters (configurable via `PINESCRIPT_CHUNK_OVERLAP`)

Each heading-based section is first identified, then recursively split if it exceeds `chunk_size`. Chunks inherit their parent section title for contextual headers.

## 6. MMR Deduplication

**File:** `rag_utils.py` — `mmr_select()`

Maximal Marginal Relevance selects documents that maximize both relevance and diversity:

```
MMR = lambda * sim(query, doc) - (1 - lambda) * max(sim(doc, selected_docs))
```

Default `lambda=0.7` (70% relevance, 30% diversity). Prevents 5 near-identical RSI chunks from dominating results.

## Configuration

All parameters are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PINESCRIPT_HYBRID_ALPHA` | 0.5 | Vector vs BM25 weight |
| `PINESCRIPT_SIMILARITY_THRESHOLD` | 0.3 | Minimum cosine similarity |
| `PINESCRIPT_RETRIEVAL_CANDIDATES` | 15 | Initial candidates before reranking |
| `PINESCRIPT_RERANK_TOP_N` | 5 | Final results after reranking |
| `PINESCRIPT_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |
| `PINESCRIPT_MMR_LAMBDA` | 0.7 | MMR relevance/diversity tradeoff |
| `PINESCRIPT_CHUNK_SIZE` | 500 | Target chunk size (characters) |
| `PINESCRIPT_CHUNK_OVERLAP` | 50 | Overlap between chunks |

## Migration

For existing databases, run the migration to add the tsvector column and update indexes:

```python
from db_schema import run_migration
await run_migration(pool)
```

This adds the `search_vector` column, populates it from existing data, creates the GIN index, and switches the HNSW index from L2 to cosine ops.

## Dependencies

Added to `requirements.txt`:
- `sentence-transformers>=3.0.0` — cross-encoder reranking
- `numpy>=1.26.0` — cosine similarity for MMR
