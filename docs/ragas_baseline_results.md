# RAGAS Baseline Results — 2026-03-06

## Configuration
- **Embedding model**: text-embedding-3-small
- **Retrieval**: L2 vector search (`<->`) — 8 results, no threshold, no reranking
- **Generation**: openai/gpt-4.1-mini (via OpenRouter)
- **Judge**: gpt-4.1-mini (via OpenAI direct)
- **DB**: VPS ragnablock_pinescript (1,148 old-style chunks, no headers, no recursive chunking)
- **Metrics**: Faithfulness (working), ContextRelevance (failed — `agenerate_text` bug in RAGAS 0.4)

## Overall Results

| Metric | Score |
|--------|-------|
| **Faithfulness** | **0.779** |
| Context Relevance | n/a (metric bug) |

## By Category

| Category | Faithfulness | Notes |
|----------|-------------|-------|
| conceptual | **0.99** | Near-perfect — LLM needs minimal context for conceptual answers |
| function_lookup | **0.84** | Good — some retrieval misses on specific function signatures |
| complex | **0.78** | Decent — drops on multi-concept queries |
| code_generation | **0.51** | Poor — LLM hallucinates code when context lacks examples |

## Per-Question Scores

### function_lookup (avg: 0.84)
| Q# | Faithfulness | Question |
|----|-------------|----------|
| 1 | 0.93 | What are the parameters of ta.ema()? |
| 2 | 1.00 | How does the ta.rsi() function work? |
| 3 | 0.71 | What does ta.macd() return? |
| 4 | 0.67 | What is the syntax for strategy.entry()? |
| 5 | 0.88 | How do I use ta.atr() to calculate ATR? |
| 6 | 0.93 | What parameters does ta.bb() accept? |
| 7 | 0.93 | How does ta.stoch() work? |
| 8 | 0.44 | What is the difference between ta.sma() and ta.ema()? |
| 9 | 1.00 | How do I use plot() to display an indicator? |
| 10 | 0.90 | What does ta.crossover() return? |

### conceptual (avg: 0.99)
| Q# | Faithfulness | Question |
|----|-------------|----------|
| 11 | 1.00 | Difference between series and simple types? |
| 12 | 1.00 | How do Pine Script variables work across bars? |
| 13 | 1.00 | Indicators vs strategies? |
| 14 | 0.97 | How does bar_index work? |
| 15 | 1.00 | Purpose of indicator() declaration? |
| 16 | 0.95 | How do request.security() calls work? |
| 17 | 1.00 | Pine Script arrays vs series? |
| 18 | 1.00 | How does PineScript handle na values? |
| 19 | 1.00 | var vs varip keywords? |
| 20 | 1.00 | How do input functions work? |

### code_generation (avg: 0.51)
| Q# | Faithfulness | Question |
|----|-------------|----------|
| 21 | 0.25 | RSI indicator with overbought/oversold levels |
| 22 | 0.92 | Bollinger Bands overlay indicator |
| 23 | **0.07** | MACD crossover strategy with entry/exit |
| 24 | 0.44 | EMA 20 and EMA 50 with crossover signals |
| 25 | 0.61 | ATR for dynamic stop loss |
| 26 | 0.96 | RSI divergence detection |
| 27 | 0.74 | Trailing stop loss strategy |
| 28 | **0.00** | VWMA indicator |
| 29 | **0.13** | Bollinger breakout strategy |
| 30 | 0.97 | Support and resistance levels |

### complex (avg: 0.78)
| Q# | Faithfulness | Question |
|----|-------------|----------|
| 31 | 0.34 | Combine RSI, MACD, and Bollinger Bands |
| 32 | 0.73 | Multi-timeframe analysis |
| 33 | 0.94 | Risk management strategy |
| 34 | 1.00 | Custom alerts with multiple indicators |
| 35 | 1.00 | PineScript v5 vs v6 differences |
| 36 | **0.08** | Adaptive moving average |
| 37 | 0.84 | Limit orders and market orders |
| 38 | 0.97 | Scoring system ranking multiple assets |
| 39 | 0.89 | Historical data limitations |
| 40 | 0.97 | Pine Script library exports |

## Bottom 5 Performers (F < 0.25)

| Rank | F | Question | Root Cause |
|------|---|----------|------------|
| 1 | 0.00 | VWMA indicator | No VWMA docs in DB — LLM hallucinated entire implementation |
| 2 | 0.07 | MACD crossover strategy | Retrieved ta.macd() reference but no strategy template — LLM fabricated entry/exit logic |
| 3 | 0.08 | Adaptive moving average | Niche topic, zero relevant chunks retrieved |
| 4 | 0.13 | Bollinger breakout strategy | Retrieved BB docs but no breakout strategy pattern |
| 5 | 0.25 | RSI overbought/oversold | Retrieved ta.rsi() but not indicator scaffolding patterns |

## Analysis

### Why code_generation scores low (0.51)
The old DB chunks are flat documentation text — they describe function signatures but rarely include full working code examples. When asked to generate code, the LLM:
1. Gets function signatures from context (faithful part)
2. Fabricates the strategy/indicator scaffolding (unfaithful part — not in context)
3. Results in low faithfulness even when the generated code is correct

### Expected Tier 1 improvements
- **Contextual Chunk Headers**: "+Section: Strategies" headers help retrieve strategy-relevant chunks
- **Recursive Chunking with Overlap**: Preserves code examples that span chunk boundaries
- **Hybrid Search (BM25+Vector)**: BM25 catches exact function names that vector search misses
- **Cross-Encoder Reranking**: Promotes chunks with actual code examples over API reference text
- **Similarity Threshold**: Drops irrelevant results that dilute context
- **Target**: code_generation 0.51 → 0.75+, overall 0.78 → 0.85+

## Known Issues
- **ContextRelevance metric failed**: `'InstructorLLM' object has no attribute 'agenerate_text'` — RAGAS 0.4 `_nv_metrics.ContextRelevance` uses async text generation interface not supported by `llm_factory()`. Needs fix before Tier 1 comparison run.
- **AnswerRelevancy dropped**: Requires `embed_query` interface not in RAGAS 0.4 `OpenAIEmbeddings`. Not fixable without custom wrapper.
