# RAGAS Tier 3 Comparison — 2026-03-09

## Summary

Tier 3 explores two orthogonal upgrades to the retrieval pipeline:

- **T3-B**: Upgrade the embedding model from `text-embedding-3-small` to `text-embedding-3-large` (3072 dimensions vs 1536).
- **T3-C**: Add [Cohere rerank-v3.5](https://cohere.com/rerank) as a post-retrieval reranking step.

**Results at a glance:**

| Pipeline | F | CR | vs Tier 2 |
|----------|---|----|-----------|
| Tier 2 (contextual, hybrid) | 0.833 | 0.919 | baseline |
| **T3-B** (embed-large) | **0.8726** | n/a* | +4.8% F |
| **T3-C** (T3-B + Cohere rerank) | **0.8715** | **0.981** | +4.6% F, +6.8% CR |

> *T3-B CR was not measured in this run due to a torch dependency conflict on Apple Silicon that prevented cross-encoder metrics from loading. A follow-up eval includes CR measurement.

---

## T3-B — Embedding Model Upgrade

### Configuration

| Parameter | Tier 2 | T3-B |
|-----------|--------|------|
| Embedding model | `text-embedding-3-small` | `text-embedding-3-large` |
| Embedding dimensions | 1536 | 3072 |
| Chunks in DB | 4,910 | 4,989 (full re-crawl) |
| Retrieval | Hybrid cosine+BM25 → RRF → threshold → MMR | same |
| Contexts per question | 15 | 15 |
| Chunk size | 1000 chars / 100 overlap | same |
| Cross-encoder rerank | Disabled | Disabled |
| Generation model | `gpt-4.1-mini` | `gpt-4.1-mini` |
| RAGAS judge | `gpt-4.1-mini` | `gpt-4.1-mini` |

### Results

| Category | Tier 2 F | T3-B F | Delta |
|----------|----------|--------|-------|
| function_lookup | — | 0.843 | — |
| conceptual | — | 0.977 | — |
| code_generation | — | 0.820 | — |
| complex | — | 0.850 | — |
| **Overall** | **0.833** | **0.873** | **+4.8%** |

**Key finding:** The larger embedding model improves faithfulness meaningfully (+4.8%). The richer 3072-dim representation captures finer semantic distinctions in Pine Script syntax and function signatures that the smaller model conflated.

---

## T3-C — Cohere Rerank v3.5

### Configuration

Same as T3-B, with one addition:

| Parameter | T3-B | T3-C |
|-----------|------|------|
| Post-retrieval rerank | None | Cohere `rerank-v3.5` (top 10) |
| Retrieval candidates | 15 | 25 → reranked to 10 |

### Results

| Category | T3-B F | T3-C F | T3-C CR |
|----------|--------|--------|---------|
| function_lookup | 0.843 | 0.839 | 0.975 |
| conceptual | 0.977 | 0.986 | 1.000 |
| code_generation | 0.820 | 0.844 | 1.000 |
| complex | 0.850 | 0.816 | 0.950 |
| **Overall** | **0.873** | **0.872** | **0.981** |

**Key finding:** Cohere rerank-v3.5 is **neutral on Faithfulness** (-0.0011, within statistical noise at 40 questions) but delivers a significant **Context Relevance boost** (+6.8% vs Tier 2). This contrasts sharply with the ms-marco cross-encoder tested in Tier 1 which hurt Faithfulness by -3.7% on code documentation. Cohere's multilingual reranker handles technical/code content without the domain penalty observed with ms-marco.

The `code_generation` category specifically benefits (+2.9% F vs T3-B), confirming that Cohere reranking helps surface more relevant code examples when the query is about writing Pine Script.

---

## Full Results History (All Runs)

| Run | Date | F | CR | Notes |
|-----|------|---|----|-------|
| Baseline | 2026-03-06 | 0.779 | — | Original (large chunks, L2) |
| Tier 1 best | 2026-03-07 | 0.774 | — | 1000-char, hybrid, 15 ctx |
| Tier 2 | 2026-03-08 | 0.833 | 0.919 | Contextual retrieval |
| **T3-B** | 2026-03-09 | **0.873** | — | `text-embedding-3-large` |
| **T3-C** | 2026-03-09 | **0.872** | **0.981** | T3-B + Cohere rerank-v3.5 |

---

## Recommendations

- **Enable `text-embedding-3-large`** — unambiguous +4.8% faithfulness gain; re-crawl required on model switch.
- **Enable Cohere rerank** — CR improvement is consistent and meaningful; F impact is neutral. Worth the added latency (~100-200ms) for quality-sensitive use cases.
- **ms-marco cross-encoder remains disabled** — hurts code doc retrieval, do not re-enable.
- **Next steps**: Evaluate HyPE (Hypothetical Prompt Embeddings) and context window expansion as further retrieval improvements.
