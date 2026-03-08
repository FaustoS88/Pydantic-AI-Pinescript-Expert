# RAG Improvements — Tier 2

Index-time LLM enrichment: code-aware chunking + Anthropic Contextual Retrieval + metadata columns.

These build on top of [Tier 1](RAG_IMPROVEMENTS.md) (hybrid search, MMR, recursive chunking, similarity
threshold). Tier 2 works at **crawl time** — every chunk gets LLM-generated context before being
embedded, so the embedding itself carries more semantic meaning.

**RAGAS result: F=0.833 (+7.6% over Tier 1 F=0.774)**

---

## Overview

| # | Improvement | Expected Impact | Status |
|---|-------------|-----------------|--------|
| 1 | Code-aware chunking | Preserves fenced code blocks intact | Implemented |
| 2 | Contextual Retrieval (Anthropic) | -49% retrieval failures (standalone) | Implemented |
| 3 | Content type detection | Enables future filtered retrieval | Implemented |
| 4 | Metadata columns (chunk_index, content_type, contextual_prefix) | Schema for future expansions | Implemented |

---

## 1. Code-Aware Chunking

**File:** `rag_utils.py` — `code_aware_split()`
**Replaces:** `recursive_character_split()` in `pinescript_recrawl_light.py`

Recursive character splitting cuts text at character boundaries — it has no awareness of code blocks
and will happily split a Pine Script example mid-line:

```pine
// chunk boundary here — example is cut in half
length = 14)
plot(ta.ema(close, length))  ← this line ends up in the next chunk
```

`code_aware_split` parses text into alternating prose/code segments by splitting on triple-backtick
boundaries, then applies different rules to each:

- **Fenced code blocks** (` ``` ... ``` `): **never split**, even if they exceed `chunk_size`. A 2,000-char
  Pine Script example stays in one chunk.
- **Prose segments**: delegate to `_rag_split_recursive` (same recursive logic as Tier 1)
- **Inline code** (single backticks): treated as prose — only fenced blocks are protected

```python
# Algorithm sketch
parts = text.split("```")
for i, part in enumerate(parts):
    if i % 2 == 0:
        segments.append(("prose", part))   # even index = prose
    else:
        segments.append(("code", f"```{part}```"))  # odd index = fenced code block
```

**Why it matters:** Pine Script docs are example-heavy. A chunked example loses all context about
what function is being demonstrated. Keeping examples whole means the embedding captures the full
pattern, not a fragment.

---

## 2. Contextual Retrieval

**File:** `rag_utils.py` — `generate_contextual_prefix()`
**Crawl flag:** `python pinescript_recrawl_light.py --contextual --clear`
**Model:** `openai/gpt-4.1-mini` via OpenRouter (configurable via `PINESCRIPT_CONTEXTUAL_MODEL`)

Implements [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
approach verbatim. Before embedding each chunk, an LLM writes a one-sentence description grounding
the chunk in its full document context.

**The problem it solves:** chunks lose context when extracted from their page. A chunk containing
*"The function returns a float"* could be about any of dozens of functions. The embedding is
ambiguous. With a contextual prefix:

```
[Context: PineScript ta.ema() returns the exponentially weighted moving average as a float series,
taking source and length parameters, located in the Technical Analysis reference.]

The function returns a float. The ta.ema() function calculates the EMA using...
```

Now the embedding knows which function this is about. Retrieval for *"ta.ema return type"* finds
this chunk instead of any generic float-returning function.

**Prompt sent to LLM (per chunk):**
```
Here is a page from the PineScript v5 documentation.
Page title: {page_title}
Page content (excerpt): {full_page_content[:8000]}

Here is a specific chunk from this page:
{chunk[:600]}

Write one concise sentence (30-80 words) describing what this chunk covers.
Be specific: name the function, concept, or language feature.
Start with "PineScript" or the function name. Reply with ONLY the sentence.
```

**What gets stored:**
- The prefix is stored separately in the `contextual_prefix` DB column for inspection
- The embedding is computed on `[Context: {prefix}]\n\n{original chunk}`
- The raw chunk text is still the content returned to the LLM at generation time

**Error handling:** if the LLM call fails, falls back to `page_title` as the prefix — no crash, no
skipped chunk.

**Benchmarks (Anthropic, Sep 2024):**
- Standalone: -49% retrieval failures
- Combined with reranking: -67% retrieval failures

**Our measured result:** +7.6% Faithfulness, +21% on code generation questions.

**Cost:** ~$8 for 4,910 chunks with `gpt-4.1-mini`. Re-crawl required to activate.

---

## 3. Content Type Detection

**File:** `rag_utils.py` — `detect_content_type()`
**Stored in:** `content_type` column (`'reference'`, `'example'`, `'tutorial'`)

A fast heuristic that classifies each chunk without an LLM call:

```python
code_blocks = content.count("```") // 2
has_param_keywords = any(k in content.lower()
    for k in ["parameter", "argument", "syntax", "returns", "return type"])
word_count = len(content.split())

if code_blocks >= 2 and word_count < 200:    → 'example'   (dense code, few words)
if has_param_keywords and code_blocks >= 1:  → 'reference' (API reference page)
else:                                         → 'tutorial'  (prose explanation)
```

**Current use:** stored for audit/inspection only.
**Future use:** `WHERE content_type = 'example'` filtered retrieval for code generation queries —
only retrieve chunks with working code examples when the user asks to generate a strategy.

---

## 4. Metadata Columns

Three new columns on `pinescript_docs`:

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `chunk_index` | `INTEGER` | `0` | Position within page — enables context window expansion (fetch neighboring chunks) |
| `content_type` | `TEXT` | `'reference'` | Heuristic type — enables filtered retrieval |
| `contextual_prefix` | `TEXT` | `NULL` | LLM-generated sentence — stored for audit; embedded at crawl time |

Migration is **idempotent** — safe to run on existing tables:

```sql
ALTER TABLE pinescript_docs ADD COLUMN IF NOT EXISTS chunk_index       INTEGER DEFAULT 0;
ALTER TABLE pinescript_docs ADD COLUMN IF NOT EXISTS content_type      TEXT    DEFAULT 'reference';
ALTER TABLE pinescript_docs ADD COLUMN IF NOT EXISTS contextual_prefix TEXT;
```

`chunk_index` specifically unlocks **context window expansion** in a future sprint: after retrieving
a chunk, fetch its neighbors (chunk_index-1, chunk_index+1) from the same page for fuller context.

---

## Running the Contextual Re-Crawl

Standard re-crawl (code-aware split only, no LLM prefix — free):
```bash
python pinescript_recrawl_light.py --clear
```

Contextual re-crawl (code-aware split + LLM prefix per chunk — ~$8, ~2h for 4,910 chunks):
```bash
python pinescript_recrawl_light.py --contextual --clear
```

Configuration via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PINESCRIPT_CONTEXTUAL_RETRIEVAL` | `false` | Enable LLM prefix at crawl time |
| `PINESCRIPT_CONTEXTUAL_MODEL` | `openai/gpt-4.1-mini` | LLM for prefix generation |
| `PINESCRIPT_MAX_PAGE_CONTEXT_CHARS` | `8000` | Page excerpt size sent to LLM |

---

## What Comes Next (Tier 3)

| # | Technique | Expected Impact |
|---|-----------|-----------------|
| 1 | **Synthetic code examples** | Complete strategy templates close the code_generation gap |
| 2 | **text-embedding-3-large** | Better code semantic understanding (+5-10% retrieval quality) |
| 3 | **Code-specific reranker** | ms-marco hurts code docs; need a programming-trained reranker |
| 4 | **HyPE** | Index-time hypothetical question generation eliminates query↔answer vocabulary gap |
| 5 | **Context window expansion** | Use chunk_index to fetch neighboring chunks after retrieval |
