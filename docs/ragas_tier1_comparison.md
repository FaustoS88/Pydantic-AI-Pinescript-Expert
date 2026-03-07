# RAGAS Tier 1 Comparison — 2026-03-06/07

## Summary

Tier 1 RAG improvements (hybrid search, contextual headers, recursive chunking, MMR dedup) were evaluated across **7 configurations** over two days. After initial regression from 500-char chunks, **1000-char chunks with top_n=15 recovered to F=0.774** — within 0.5% of the original baseline (0.779).

**Key finding:** Cross-encoder reranking with ms-marco model **hurts** faithfulness (-3.7%) on code/technical docs. A domain-specific reranker is needed for Tier 2.

**Best config:** 1000-char chunks, 15 contexts, hybrid search + MMR, NO cross-encoder reranking.

## Configuration Matrix

| Parameter | Baseline (old) | T1 (old) | Base (500) | T1 (500, 5) | T1 (500, 15) | **T1 (1000, 15)** | T1 (1000, rerank) |
|-----------|---------------|----------|------------|-------------|--------------|-------------------|-------------------|
| Chunks in DB | 1,148 (flat) | 1,148 (flat) | 9,978 (hdr) | 9,978 (hdr) | 9,978 (hdr) | **4,911 (hdr)** | 4,911 (hdr) |
| Avg chunk size | 2,452 chars | 2,452 chars | 460 chars | 460 chars | 460 chars | **835 chars** | 835 chars |
| Retrieval | L2 vector | Hybrid+MMR | L2 vector | Hybrid+MMR | Hybrid+MMR | **Hybrid+MMR** | Hybrid+Rerank+MMR |
| Contexts/question | 8 | 5 | 8 | 5 | 15 | **15** | 15 |
| Est. context chars | ~19,600 | ~12,260 | ~3,680 | ~2,300 | ~6,900 | **~12,525** | ~12,525 |
| Chunk headers | No | No | Yes | Yes | Yes | **Yes** | Yes |
| Recursive chunking | No | No | Yes (500/50) | Yes (500/50) | Yes (500/50) | **Yes (1000/100)** | Yes (1000/100) |
| Similarity threshold | None | 0.3 | None | 0.3 | 0.3 | **0.3** | 0.3 |
| Cross-encoder rerank | Off | Off | Off | Off | Off | **Off** | **On (ms-marco)** |
| Embedding model | text-embedding-3-small | same | same | same | same | same | same |
| Generation model | gpt-4.1-mini | same | same | same | same | same | same |

## Overall Results

| Run | Faithfulness | Delta vs Baseline | Notes |
|-----|-------------|-------------------|-------|
| **Baseline (old chunks, 8 ctx)** | **0.779** | — | Original baseline |
| Tier 1 (old chunks, 5 ctx) | 0.677 | -13.1% | Fewer contexts hurt |
| Baseline (500-char, 8 ctx) | 0.621 | -20.3% | Smaller chunks = less context |
| Tier 1 (500-char, 5 ctx) | 0.561 | -28.0% | Worst: small chunks + few ctx |
| Tier 1 (500-char, 15 ctx) | 0.711 | -8.7% | More contexts recover most loss |
| **Tier 1 (1000-char, 15 ctx)** | **0.774** | **-0.6%** | **Best Tier 1 — matches baseline** |
| Tier 1 (1000-char, rerank) | 0.745 | -4.4% | Cross-encoder hurts on code docs |

## By Category

| Category | Base (old) | T1 (old) | Base (500) | T1 (500,5) | T1 (500,15) | **T1 (1000,15)** | T1 (rerank) |
|----------|-----------|----------|------------|------------|-------------|-------------------|-------------|
| conceptual | **0.99** | 0.97 | 0.95 | 0.87 | 0.99 | **0.98** | 0.96 |
| function_lookup | **0.84** | 0.69 | 0.54 | 0.50 | 0.68 | **0.83** | 0.77 |
| complex | **0.78** | 0.69 | 0.58 | 0.56 | 0.73 | **0.75** | 0.75 |
| code_generation | **0.51** | 0.35 | 0.41 | 0.32 | 0.44 | **0.53** | 0.49 |

## Root Cause Analysis

### Why 500-char Chunks Regressed

1. Old chunks averaged **2,452 chars** — substantial context per chunk
2. 500-char chunks average **460 chars** — 5.3x smaller
3. With 8 contexts: old = ~19,600 chars, new = ~3,680 chars → **5.3x less material**
4. Less material → LLM fills gaps with hallucination → faithfulness drops

### Why 1000-char Chunks Recover

1. 1000-char chunks average **835 chars** (with headers)
2. 15 contexts × 835 chars = **~12,525 chars** — approaching baseline's ~19,600
3. Hybrid search + MMR provides better diversity than L2 alone
4. Contextual headers help retrieval accuracy

### Why Cross-Encoder Reranking Hurts (-3.7%)

1. `ms-marco-MiniLM-L-6-v2` trained on **web search** queries, not code/technical docs
2. It promotes "textual fluency" over code-relevant passages
3. Function signatures and code examples get demoted in favor of prose
4. **Fix for Tier 2**: Use a code-specific reranker or fine-tune on programming Q&A pairs

### What the Retrieval Pipeline Gets Right

Context Relevance scores (where measured) remain good:
- **conceptual**: 0.85–1.00
- **complex**: 0.82–0.85
- **function_lookup**: 0.78
- **code_generation**: 0.75

The retrieval pipeline finds relevant results. The remaining gap is in **what the docs contain** — function signatures but few working code examples.

## Recovery Journey

| Step | Change | Faithfulness | Delta |
|------|--------|-------------|-------|
| Start | 500-char, 5 ctx | 0.561 | -28.0% |
| +contexts | 500-char, 15 ctx | 0.711 | -8.7% |
| +chunk size | 1000-char, 15 ctx | **0.774** | **-0.6%** |
| +reranking | 1000-char, rerank | 0.745 | -4.4% (worse!) |

## Recommended Next Steps (Revised)

### Tier 1 Complete — Current Config
- **CHUNK_SIZE=1000, CHUNK_OVERLAP=100, RERANK_TOP_N=15**
- Cross-encoder: available but disabled (ms-marco hurts code docs)
- This is the production config for RuneForge RAG

### Tier 2 (Medium-term)
1. **Semantic chunking** — respect code block boundaries, never split examples
2. **Synthetic code examples** — use RuneForge to generate + backtest canonical strategies, embed as "golden examples"
3. **text-embedding-3-large** — better embeddings for code semantics
4. **Code-specific reranker** — train/select for programming Q&A (not ms-marco)
5. **Port RAGAS to Ragnarok** — RuneForge/Mimir/Odin RAG testing

## Crawl Statistics

| Metric | Old DB | 500-char DB | 1000-char DB |
|--------|--------|-------------|--------------|
| Pages crawled | 64 | 75 | 75 |
| Total chunks | 1,148 | 9,978 | **4,911** |
| Avg chunk size | 2,452 chars | 460 chars | **835 chars** |
| Contextual headers | 0% | 100% | **100%** |
| Chunking strategy | Flat (heading) | Recursive (500/50) | **Recursive (1000/100)** |
| tsvector (BM25) | Populated | Populated | **Populated** |
| Crawler | Crawl4AI (Chromium) | httpx+html2text | **httpx+html2text** |

## Cross-Encoder Fix (2026-03-07)

torch 2.10.0 caused SIGBUS on Apple Silicon during BERT inference. Fixed by downgrading:
- torch: 2.10.0 → 2.6.0
- torchvision: 0.25.0 → 0.21.0
- av: rebuilt from source (broken .so files)

Cross-encoder works but ms-marco model is unsuitable for code docs.

## Files

- Baseline (old): `results/baseline_20260306.json`
- Tier 1 (old chunks): `results/tier1_20260306.json`
- Baseline (500-char): `results/baseline_recrawl_20260306.json`
- Tier 1 (500-char, 5 ctx): `results/tier1_recrawl_20260306.json`
- Tier 1 (500-char, 15 ctx): `results/tier1_recrawl_top15_20260306.json`
- **Tier 1 (1000-char, 15 ctx)**: `results/tier1_1000chunk_top15_20260307.json`
- **Tier 1 (1000-char, rerank)**: `results/tier1_1000chunk_rerank_20260307.json`
- Harness: `tests/ragas_eval.py`
- Lightweight crawler: `pinescript_recrawl_light.py`
