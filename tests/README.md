# RAGAS Evaluation — PineScript Expert

Measures RAG quality (faithfulness, answer relevancy, context precision) using the
[RAGAS](https://docs.ragas.io/) framework as an LLM-as-judge.

---

## Prerequisites

### 1. Python dependencies

```bash
pip install ragas datasets langchain-openai
```

RAGAS requires:
- `ragas` — evaluation framework
- `datasets` — HuggingFace data structures used internally by RAGAS
- `langchain-openai` — RAGAS uses LangChain's OpenAI wrapper for judge LLM calls

### 2. API keys

```bash
# Real OpenAI key (NOT OpenRouter) — used for embeddings AND RAGAS judge calls
export OPENAI_API_KEY=sk-...

# Optional: OpenRouter key for the generation model
export OPENROUTER_API_KEY=sk-or-...
```

### 3. Database connection

The evaluation connects to the `pinescript_docs` PostgreSQL/pgvector database.

**Option A — VPS SSH tunnel (recommended for baseline):**

```bash
# Open tunnel in a separate terminal and leave it running
ssh -L 5433:localhost:55432 root@<your-vps-host>

# Then set DATABASE_URL in your shell
export DATABASE_URL=postgresql://<user>:<password>@localhost:5433/<dbname>
```

**Option B — Local Docker:**

```bash
docker run -d --name pinescript-db \
  -e POSTGRES_USER=pine_user \
  -e POSTGRES_PASSWORD=pine_pass \
  -e POSTGRES_DB=pinescript_docs \
  -p 5433:5432 \
  pgvector/pgvector:pg16

export DATABASE_URL=postgresql://pine_user:pine_pass@localhost:5433/pinescript_docs

# Populate with the crawler (if empty)
python pinescript_crawler.py
```

---

## Running the evaluation

```bash
# From project root
python tests/ragas_eval.py

# With a custom output path
python tests/ragas_eval.py --output results/baseline_20260306.json
```

The harness will:
1. Load the 40 questions from `tests/ragas_testset.json`
2. For each question: embed → vector search → retrieve top-8 context chunks
3. Call the agent to generate an answer using those contexts
4. Pass questions + answers + contexts to RAGAS `evaluate()`
5. Print a summary report and save full JSON to `results/`

**Expected runtime:** ~10–20 minutes for 40 questions (LLM calls + RAGAS judge calls).

**Expected cost:** ~$0.50 per full run (RAGAS judge calls via OpenAI).

---

## Interpreting results

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Faithfulness** | Is the answer grounded in the retrieved context? (no hallucination) | > 0.85 |
| **Answer Relevancy** | Does the answer actually address the question? | > 0.80 |
| **Context Precision** | Were the retrieved chunks relevant to the question? | > 0.90 |

Score range: 0.0 – 1.0 (higher is better).

### Categories

| Category | Description | Expected difficulty |
|----------|-------------|-------------------|
| `function_lookup` | Direct questions about specific functions | Easiest — exact match |
| `conceptual` | Language concepts and types | Medium |
| `code_generation` | Questions requiring code synthesis | Hard |
| `complex` | Multi-hop questions combining multiple topics | Hardest |

### Low scores indicate

- **Low Faithfulness** → Agent is hallucinating beyond retrieved context
- **Low Context Precision** → Vector search is returning irrelevant chunks (embedding or chunking issue)
- **Low Answer Relevancy** → Answer is off-topic or too generic (LLM or prompt issue)

---

## Test question set

`tests/ragas_testset.json` contains 40 questions:
- 10 function lookups (`ta.ema()`, `ta.rsi()`, `strategy.entry()`, etc.)
- 10 conceptual (`series vs simple types`, `var vs varip`, `na handling`, etc.)
- 10 code generation (RSI indicator, MACD strategy, ATR stop loss, etc.)
- 10 complex/multi-hop (multi-timeframe, risk management, v5→v6 differences, etc.)

---

## Results files

Results are saved to `results/` (gitignored). Each run produces a JSON file:

```
results/
├── baseline_20260306.json    # before any RAG improvements
├── headers_20260307.json     # after adding contextual chunk headers
├── hybrid_20260307.json      # after adding BM25 hybrid search
└── ...
```

The JSON structure:

```json
{
  "meta": { "date": "...", "questions": 40, "embedding_model": "...", ... },
  "results": {
    "overall": { "faithfulness": 0.xx, "answer_relevancy": 0.xx, "context_precision": 0.xx },
    "by_category": { "function_lookup": { ... }, ... },
    "worst_performers": [ ... ],
    "per_question": [ { "question": "...", "answer": "...", "faithfulness": 0.xx, ... } ]
  }
}
```
