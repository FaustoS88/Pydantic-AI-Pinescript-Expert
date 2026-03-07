"""RAGAS evaluation harness for the PineScript Expert RAG pipeline.

Usage:
    python tests/ragas_eval.py [--output results/baseline_YYYYMMDD.json] [--retrieval baseline|tier1]

    --retrieval baseline  L2 vector search, no threshold, no reranking (default)
    --retrieval tier1     Hybrid cosine+BM25, RRF, similarity threshold, cross-encoder rerank, MMR

Prerequisites:
    pip install ragas datasets langchain-openai
    OPENAI_API_KEY env var (real OpenAI key, not OpenRouter)
    OPENROUTER_API_KEY env var (for answer generation)
    DATABASE_URL pointing to the pinescript_docs database
    (or VPS tunnel: ssh -L 5433:localhost:55432 root@<your-vps-host>)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import pydantic_core
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ragas_eval")

# Resolve project root so we can import config regardless of cwd
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_DATABASE_URL,
    EMBEDDING_MODEL,
    VECTOR_SEARCH_LIMIT,
    HYBRID_SEARCH_ALPHA,
    SIMILARITY_THRESHOLD,
    RETRIEVAL_CANDIDATES,
    RERANK_TOP_N,
    MMR_LAMBDA,
)
from rag_utils import hybrid_retrieve  # noqa: E402

TESTSET_PATH = SCRIPT_DIR / "ragas_testset.json"
RESULTS_DIR = PROJECT_ROOT / "results"

CATEGORIES = ["function_lookup", "conceptual", "code_generation", "complex"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "openai/gpt-4.1-mini")

SYSTEM_PROMPT = (
    "You are a Pine Script v6 expert assistant. Pine Script is the programming "
    "language used in TradingView for creating custom indicators and strategies. "
    "Answer based on the retrieved documentation context. Include code examples "
    "when relevant. Be practical and give working solutions."
)

# ---------------------------------------------------------------------------
# Retrieval — baseline (L2) or tier1 (hybrid cosine+BM25, rerank, MMR)
# ---------------------------------------------------------------------------

async def retrieve_contexts_baseline(
    question: str,
    openai_client: AsyncOpenAI,
    pool: asyncpg.Pool,
    limit: int = VECTOR_SEARCH_LIMIT,
) -> list[str]:
    """Baseline: L2 vector search, no threshold, no reranking."""
    resp = await openai_client.embeddings.create(
        input=question,
        model=EMBEDDING_MODEL,
    )
    embedding_vector = resp.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding_vector).decode()

    rows = await pool.fetch(
        f"SELECT url, title, content FROM pinescript_docs "
        f"ORDER BY embedding <-> $1 LIMIT {limit}",
        embedding_json,
    )
    if not rows:
        return []
    return [
        f"# {row['title']}\nURL: {row['url']}\n\n{row['content']}"
        for row in rows
    ]


async def retrieve_contexts_tier1(
    question: str,
    openai_client: AsyncOpenAI,
    pool: asyncpg.Pool,
) -> list[str]:
    """Tier 1: hybrid cosine+BM25 → RRF → threshold → MMR.

    Cross-encoder reranking available but disabled — ms-marco model hurts
    faithfulness on code/technical docs (-3.7%). A code-specific reranker
    is needed for Tier 2.
    """
    docs = await hybrid_retrieve(
        pool=pool,
        openai_client=openai_client,
        query=question,
        embedding_model=EMBEDDING_MODEL,
        candidates=RETRIEVAL_CANDIDATES,
        alpha=HYBRID_SEARCH_ALPHA,
        threshold=SIMILARITY_THRESHOLD,
        top_n=RERANK_TOP_N,
        mmr_lambda=MMR_LAMBDA,
        enable_rerank=False,
    )
    if not docs:
        return []
    return [
        f"# {doc.title}\nURL: {doc.url}\n\n{doc.content}"
        for doc in docs
    ]


# ---------------------------------------------------------------------------
# Generation — direct OpenRouter call (bypasses pydantic-ai version issues)
# ---------------------------------------------------------------------------

async def generate_answer(
    question: str,
    contexts: list[str],
    http_client: httpx.AsyncClient,
) -> str:
    """Call OpenRouter directly with retrieved contexts and return the answer."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        return "[ERROR] OPENROUTER_API_KEY not set"

    context_block = "\n\n---\n\n".join(contexts) if contexts else "(no context retrieved)"

    user_msg = (
        f"## Retrieved Documentation\n\n{context_block}\n\n"
        f"---\n\n## Question\n\n{question}"
    )

    try:
        resp = await http_client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        logger.error("Generation error for question %r: %s", question[:60], exc)
        return f"[ERROR] {exc}"


# ---------------------------------------------------------------------------
# RAGAS evaluation (v0.4 API)
# ---------------------------------------------------------------------------

def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
) -> dict[str, Any]:
    """Run RAGAS evaluate() and return a plain dict of scores.

    Uses dual-LLM approach:
    - InstructorLLM (llm_factory) for Faithfulness — uses agenerate() interface
    - LangchainLLMWrapper for ContextRelevance — uses agenerate_text() interface

    Both are old-style metrics compatible with evaluate().
    """
    try:
        from ragas import evaluate, RunConfig
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._nv_metrics import ContextRelevance
        from ragas.llms import llm_factory, LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        from openai import OpenAI
    except ImportError as exc:
        logger.error(
            "RAGAS dependencies missing. Run: pip install ragas datasets langchain-openai\n%s",
            exc,
        )
        return {}

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)

    # InstructorLLM for Faithfulness (uses agenerate interface)
    instructor_llm = llm_factory("gpt-4.1-mini", client=openai_client)
    instructor_llm.model_args["max_tokens"] = 8192

    # LangchainLLMWrapper for ContextRelevance (uses agenerate_text interface)
    langchain_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4.1-mini", api_key=openai_api_key)
    )

    metrics = [
        Faithfulness(llm=instructor_llm),
        ContextRelevance(llm=langchain_llm),
    ]

    # Build evaluation dataset
    samples = []
    for q, a, ctx in zip(questions, answers, contexts):
        samples.append(SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=ctx,
        ))
    dataset = EvaluationDataset(samples=samples)

    run_config = RunConfig(max_retries=2, max_wait=120, max_workers=4)

    logger.info("Running RAGAS evaluate() on %d samples (2 metrics x %d) …", len(questions), len(questions))
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    # Convert to plain dict
    try:
        scores_df = result.to_pandas()
        return scores_df.to_dict(orient="list")
    except Exception:  # noqa: BLE001
        return dict(result)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

METRIC_KEYS = ["faithfulness", "nv_context_relevance"]
DISPLAY_NAMES = {
    "faithfulness": "Faithfulness",
    "nv_context_relevance": "Context Relevance (NV)",
}


def _safe_avg(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def build_report(
    items: list[dict[str, Any]],
    ragas_scores: dict[str, Any],
) -> dict[str, Any]:
    """Merge per-question RAGAS scores back into items and compute aggregates."""
    for i, item in enumerate(items):
        for k in METRIC_KEYS:
            raw = ragas_scores.get(k)
            item[k] = raw[i] if raw and i < len(raw) else None

    # Aggregate overall
    overall: dict[str, Any] = {}
    for k in METRIC_KEYS:
        vals = [it.get(k) for it in items]
        overall[k] = _safe_avg(vals)

    # Per-category
    by_category: dict[str, dict[str, Any]] = {}
    for cat in CATEGORIES:
        cat_items = [it for it in items if it["category"] == cat]
        cat_scores: dict[str, Any] = {}
        for k in METRIC_KEYS:
            vals = [it.get(k) for it in cat_items]
            cat_scores[k] = _safe_avg(vals)
        by_category[cat] = cat_scores

    # Bottom-5 by context relevance
    cp_key = "nv_context_relevance"
    sortable = [it for it in items if it.get(cp_key) is not None]
    worst = sorted(sortable, key=lambda x: x[cp_key])[:5]

    return {
        "overall": overall,
        "by_category": by_category,
        "worst_performers": [
            {
                "question": it["question"],
                "faithfulness": it.get("faithfulness"),
                "context_precision": it.get(cp_key),
                "answer_relevancy": it.get("answer_relevancy"),
            }
            for it in worst
        ],
        "per_question": items,
    }


def print_report(report: dict[str, Any], meta: dict[str, Any]) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("RAGAS Evaluation")
    print(sep)
    for k, v in meta.items():
        print(f"{k}: {v}")

    print("\nOverall:")
    for k, v in report["overall"].items():
        label = DISPLAY_NAMES.get(k, k.replace("_", " ").title())
        val = f"{v:.4f}" if v is not None else "n/a"
        print(f"  {label:<22} {val}")

    print("\nBy Category:")
    for cat, scores in report["by_category"].items():
        f_ = scores.get("faithfulness")
        cr = scores.get("nv_context_relevance")
        f_s = f"{f_:.2f}" if f_ is not None else "n/a"
        cr_s = f"{cr:.2f}" if cr is not None else "n/a"
        print(f"  {cat:<22} F={f_s}  CR={cr_s}")

    if report["worst_performers"]:
        print("\nWorst performers (bottom 5 by context_precision):")
        for it in report["worst_performers"]:
            cp = it.get("context_precision")
            f_ = it.get("faithfulness")
            cp_s = f"{cp:.2f}" if cp is not None else "n/a"
            f_s = f"{f_:.2f}" if f_ is not None else "n/a"
            print(f"  Q: \"{it['question'][:70]}\" → CP={cp_s}, F={f_s}")

    print(sep)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

RETRIEVAL_METHODS = {
    "baseline": "L2 vector search (baseline — no threshold, no reranking)",
    "tier1": "Hybrid cosine+BM25 → RRF → threshold → MMR (rerank disabled — torch conflict)",
}


async def run_evaluation(output_path: Path, retrieval_mode: str = "baseline") -> None:
    with open(TESTSET_PATH) as fh:
        testset = json.load(fh)
    logger.info("Loaded %d questions from %s", len(testset), TESTSET_PATH)
    logger.info("Retrieval mode: %s", retrieval_mode)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY is not set. Aborting.")
        sys.exit(1)

    db_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    logger.info("Connecting to database: %s", db_url.split("@")[-1])

    openai_client = AsyncOpenAI(api_key=openai_api_key)

    try:
        pool = await asyncpg.create_pool(db_url)
    except Exception as exc:
        logger.error(
            "Cannot connect to database. "
            "If using VPS, open an SSH tunnel first:\n"
            "  ssh -L 5433:localhost:55432 root@<your-vps-host>\n"
            "Then set: DATABASE_URL=postgresql://<user>:<pass>@localhost:5433/<dbname>\n"
            "Error: %s",
            exc,
        )
        sys.exit(1)

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    items: list[dict[str, Any]] = []

    retrieve_fn = retrieve_contexts_tier1 if retrieval_mode == "tier1" else retrieve_contexts_baseline

    async with httpx.AsyncClient() as http_client:
        try:
            for i, entry in enumerate(testset, 1):
                q = entry["question"]
                cat = entry["category"]
                logger.info("[%d/%d] %s — %s", i, len(testset), cat, q[:60])

                if retrieval_mode == "tier1":
                    ctx_chunks = await retrieve_fn(q, openai_client, pool)
                else:
                    ctx_chunks = await retrieve_fn(q, openai_client, pool)
                answer = await generate_answer(q, ctx_chunks, http_client)

                questions.append(q)
                answers.append(answer)
                contexts.append(ctx_chunks)
                items.append(
                    {
                        "id": entry.get("id", i),
                        "question": q,
                        "category": cat,
                        "answer": answer,
                        "num_contexts": len(ctx_chunks),
                    }
                )
        finally:
            await pool.close()

    # RAGAS evaluation
    ragas_scores = run_ragas(questions, answers, contexts)

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta = {
        "date": run_date,
        "questions": len(testset),
        "embedding_model": EMBEDDING_MODEL,
        "retrieval_limit": VECTOR_SEARCH_LIMIT if retrieval_mode == "baseline" else RERANK_TOP_N,
        "retrieval_method": RETRIEVAL_METHODS[retrieval_mode],
        "generation_model": OPENROUTER_MODEL,
        "db_url": db_url.split("@")[-1],  # mask credentials
    }

    report = build_report(items, ragas_scores)

    output: dict[str, Any] = {
        "meta": meta,
        "results": report,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    print_report(report, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the PineScript Expert RAG pipeline.")
    parser.add_argument(
        "--retrieval",
        choices=["baseline", "tier1"],
        default="baseline",
        help="Retrieval method: baseline (L2) or tier1 (hybrid+rerank+MMR)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the JSON results file (auto-generated if omitted)",
    )
    args = parser.parse_args()

    if args.output is None:
        tag = args.retrieval
        args.output = RESULTS_DIR / f"{tag}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"

    asyncio.run(run_evaluation(args.output, retrieval_mode=args.retrieval))


if __name__ == "__main__":
    main()
