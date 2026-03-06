"""RAGAS evaluation harness for the PineScript Expert RAG pipeline.

Usage:
    python tests/ragas_eval.py [--output results/baseline_YYYYMMDD.json]

Prerequisites:
    pip install ragas datasets langchain-openai
    OPENAI_API_KEY env var (real OpenAI key, not OpenRouter)
    DATABASE_URL pointing to the pinescript_docs database
    (or VPS tunnel: ssh -L 5433:localhost:55432 root@167.235.150.184)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg
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

# Resolve project root so we can import agent.py regardless of cwd
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_DATABASE_URL,
    EMBEDDING_MODEL,
    VECTOR_SEARCH_LIMIT,
)

TESTSET_PATH = SCRIPT_DIR / "ragas_testset.json"
RESULTS_DIR = PROJECT_ROOT / "results"

CATEGORIES = ["function_lookup", "conceptual", "code_generation", "complex"]

# ---------------------------------------------------------------------------
# Retrieval — mirror of agent.py:retrieve() but returns raw list[str]
# ---------------------------------------------------------------------------

async def retrieve_contexts(
    question: str,
    openai_client: AsyncOpenAI,
    pool: asyncpg.Pool,
    limit: int = VECTOR_SEARCH_LIMIT,
) -> list[str]:
    """Embed *question* and return top-k doc chunks from the DB as strings."""
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


# ---------------------------------------------------------------------------
# Generation — thin wrapper around run_agent
# ---------------------------------------------------------------------------

async def generate_answer(question: str, openai_client: AsyncOpenAI, pool: asyncpg.Pool) -> str:
    """Call the agent and return its text answer."""
    from agent import pinescript_agent, Dependencies, create_openrouter_model

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    use_openrouter = bool(openrouter_api_key)

    deps = Dependencies(
        openai=openai_client,
        pool=pool,
        openrouter_api_key=openrouter_api_key,
        use_openrouter=use_openrouter,
    )

    try:
        if use_openrouter:
            or_model = create_openrouter_model()
            if or_model:
                with pinescript_agent.override(model=or_model):
                    result = await pinescript_agent.run(question, deps=deps)
            else:
                result = await pinescript_agent.run(question, deps=deps)
        else:
            result = await pinescript_agent.run(question, deps=deps)

        return result.data.response
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent error for question %r: %s", question[:60], exc)
        return f"[ERROR] {exc}"


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
) -> dict[str, Any]:
    """Run RAGAS evaluate() and return a plain dict of scores."""
    try:
        from ragas import evaluate  # type: ignore[import]
        from ragas.metrics import (  # type: ignore[import]
            faithfulness,
            answer_relevancy,
            context_precision,
        )
        from datasets import Dataset  # type: ignore[import]
    except ImportError as exc:
        logger.error(
            "RAGAS dependencies missing. Run: pip install ragas datasets langchain-openai\n%s",
            exc,
        )
        return {}

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )

    logger.info("Running RAGAS evaluate() on %d samples …", len(questions))
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    # result is a ragas EvaluationResult; convert to plain dict
    try:
        scores_df = result.to_pandas()
        return scores_df.to_dict(orient="list")
    except Exception:  # noqa: BLE001
        return dict(result)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _safe_avg(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def build_report(
    items: list[dict[str, Any]],
    ragas_scores: dict[str, Any],
) -> dict[str, Any]:
    """Merge per-question RAGAS scores back into items and compute aggregates."""
    # Attach per-question RAGAS scores (lists indexed by position)
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision"]
    for i, item in enumerate(items):
        for k in metric_keys:
            raw = ragas_scores.get(k)
            item[k] = raw[i] if raw and i < len(raw) else None

    # Aggregate overall
    overall: dict[str, Any] = {}
    for k in metric_keys:
        vals = [it.get(k) for it in items]
        overall[k] = _safe_avg(vals)

    # Per-category
    by_category: dict[str, dict[str, Any]] = {}
    for cat in CATEGORIES:
        cat_items = [it for it in items if it["category"] == cat]
        cat_scores: dict[str, Any] = {}
        for k in metric_keys:
            vals = [it.get(k) for it in cat_items]
            cat_scores[k] = _safe_avg(vals)
        by_category[cat] = cat_scores

    # Bottom-5 by context_precision
    sortable = [it for it in items if it.get("context_precision") is not None]
    worst = sorted(sortable, key=lambda x: x["context_precision"])[:5]

    return {
        "overall": overall,
        "by_category": by_category,
        "worst_performers": [
            {
                "question": it["question"],
                "faithfulness": it.get("faithfulness"),
                "context_precision": it.get("context_precision"),
                "answer_relevancy": it.get("answer_relevancy"),
            }
            for it in worst
        ],
        "per_question": items,
    }


def print_report(report: dict[str, Any], meta: dict[str, Any]) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("RAGAS Baseline Evaluation")
    print(sep)
    for k, v in meta.items():
        print(f"{k}: {v}")

    print("\nOverall:")
    for k, v in report["overall"].items():
        label = k.replace("_", " ").title()
        val = f"{v:.4f}" if v is not None else "n/a"
        print(f"  {label:<22} {val}")

    print("\nBy Category:")
    for cat, scores in report["by_category"].items():
        f_ = scores.get("faithfulness")
        ar = scores.get("answer_relevancy")
        cp = scores.get("context_precision")
        f_s = f"{f_:.2f}" if f_ is not None else "n/a"
        ar_s = f"{ar:.2f}" if ar is not None else "n/a"
        cp_s = f"{cp:.2f}" if cp is not None else "n/a"
        print(f"  {cat:<22} F={f_s}  AR={ar_s}  CP={cp_s}")

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

async def run_evaluation(output_path: Path) -> None:
    # Load test questions
    with open(TESTSET_PATH) as fh:
        testset = json.load(fh)
    logger.info("Loaded %d questions from %s", len(testset), TESTSET_PATH)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY is not set. Aborting.")
        sys.exit(1)

    db_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    logger.info("Connecting to database: %s", db_url)

    openai_client = AsyncOpenAI(api_key=openai_api_key)

    try:
        pool = await asyncpg.create_pool(db_url)
    except Exception as exc:
        logger.error(
            "Cannot connect to database (%s). "
            "If using VPS, open an SSH tunnel first:\n"
            "  ssh -L 5433:localhost:55432 root@167.235.150.184\n"
            "Then set: DATABASE_URL=postgresql://ragna_admin:<pass>@localhost:5433/ragnablock_pinescript\n"
            "Error: %s",
            db_url,
            exc,
        )
        sys.exit(1)

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    items: list[dict[str, Any]] = []

    try:
        for i, entry in enumerate(testset, 1):
            q = entry["question"]
            cat = entry["category"]
            logger.info("[%d/%d] %s — %s", i, len(testset), cat, q[:60])

            ctx_chunks = await retrieve_contexts(q, openai_client, pool)
            answer = await generate_answer(q, openai_client, pool)

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

    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    meta = {
        "date": run_date,
        "questions": len(testset),
        "embedding_model": EMBEDDING_MODEL,
        "retrieval_limit": VECTOR_SEARCH_LIMIT,
        "db_url": db_url.split("@")[-1],  # mask credentials
    }

    report = build_report(items, ragas_scores)

    output: dict[str, Any] = {
        "meta": meta,
        "results": report,
    }

    # Save to file
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    print_report(report, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the PineScript Expert RAG pipeline.")
    default_output = RESULTS_DIR / f"baseline_{datetime.utcnow().strftime('%Y%m%d')}.json"
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Path for the JSON results file (default: {default_output})",
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.output))


if __name__ == "__main__":
    main()
