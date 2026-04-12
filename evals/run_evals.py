"""
Cartograph Eval Runner
======================

Runs the research pipeline against the golden set and scores each result.

Usage
-----
    # Run all queries (deterministic checks only, fast)
    PYTHONPATH=. python evals/run_evals.py --no-llm-judge

    # Run all queries with LLM judge (requires Ollama)
    PYTHONPATH=. python evals/run_evals.py

    # Run a specific query by id
    PYTHONPATH=. python evals/run_evals.py --ids mrna_vaccines gut_microbiome_mental_health

    # Override model
    PYTHONPATH=. python evals/run_evals.py --model llama3.2

Results are saved to evals/results/eval_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evals.judges import deterministic, llm_judge
from src.agent.graph import build_graph
from src.config import AgentConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_runner")

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"

DEPTH_MAP = {
    "Quick (3)": 3,
    "Standard (5)": 5,
    "Deep (7)": 7,
}

# Symbols
PASS = "✅"
FAIL = "✗ "
WARN = "⚠️ "
INFO = "ℹ️ "

SEV_SYMBOL = {"error": FAIL, "warning": WARN, "info": INFO}


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(query: str, depth: str, model: str) -> tuple[dict, float]:
    """Invoke the full research pipeline. Returns (final_state, elapsed_seconds)."""
    cfg = AgentConfig(
        model_name=model,
        max_sub_questions=DEPTH_MAP[depth],
    )
    graph = build_graph(cfg)
    initial_state = {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "evaluated_sources": [],
        "coverage_sufficient": False,
        "retry_count": 0,
        "report": "",
    }
    t0 = time.monotonic()
    result = graph.invoke(initial_state)
    elapsed = time.monotonic() - t0
    return result, elapsed


# ── Result formatting ─────────────────────────────────────────────────────────

def _pipeline_stats(state: dict) -> dict:
    return {
        "sub_question_count": len(state.get("sub_questions", [])),
        "search_result_count": len(state.get("search_results", [])),
        "evaluated_source_count": len(state.get("evaluated_sources", [])),
        "retry_count": state.get("retry_count", 0),
        "report_word_count": len(state.get("report", "").split()),
        "coverage_sufficient": state.get("coverage_sufficient", False),
    }


def _overall_pass(det_results, judge_scores, min_judge_score: int = 3) -> bool:
    """PASS if no deterministic errors AND all LLM scores ≥ min_judge_score."""
    det_pass = all(
        r.passed or r.severity != "error" for r in det_results
    )
    judge_pass = all(
        s.score == 0 or s.score >= min_judge_score for s in judge_scores
    )
    return det_pass and judge_pass


# ── Console output ────────────────────────────────────────────────────────────

def _print_result(golden: dict, state: dict, det_results, judge_scores, elapsed: float) -> None:
    stats = _pipeline_stats(state)
    passed = _overall_pass(det_results, judge_scores)
    verdict = f"{PASS} PASS" if passed else f"{FAIL}FAIL"

    print(f"\n{'─' * 70}")
    print(f"  [{golden['id']}]  {golden['query'][:60]}")
    print(f"  Depth: {golden['depth']}  |  Elapsed: {elapsed:.1f}s  |  {verdict}")
    print(f"  Sub-questions: {stats['sub_question_count']}  |  "
          f"Sources: {stats['evaluated_source_count']}  |  "
          f"Retries: {stats['retry_count']}  |  "
          f"Words: {stats['report_word_count']}")
    print()

    print("  Deterministic checks:")
    for r in det_results:
        if r.severity == "info" and r.passed:
            continue  # suppress passing info-level checks for brevity
        sym = PASS if r.passed else SEV_SYMBOL.get(r.severity, FAIL)
        print(f"    {sym}  {r.message}")

    if judge_scores:
        print("\n  LLM judge scores:")
        for s in judge_scores:
            if s.score == 0:
                print(f"    ⚡  {s.dimension}: SKIPPED — {s.reason}")
            else:
                bar = "█" * s.score + "░" * (s.max_score - s.score)
                sym = PASS if s.score >= 3 else WARN if s.score == 2 else FAIL
                print(f"    {sym}  {s.dimension}: {s.score}/{s.max_score}  [{bar}]  {s.reason}")


def _print_summary(results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print(f"\n{'═' * 70}")
    print(f"  EVAL SUMMARY  —  {passed}/{total} passed")
    if failed:
        print("  Failed queries:")
        for r in results:
            if not r["passed"]:
                print(f"    • {r['id']}: {r['golden']['query'][:60]}")
    print(f"{'═' * 70}\n")


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _serialise_det(results) -> list[dict]:
    return [
        {"name": r.name, "passed": r.passed, "message": r.message, "severity": r.severity}
        for r in results
    ]


def _serialise_judge(scores) -> list[dict]:
    return [
        {"dimension": s.dimension, "score": s.score, "max_score": s.max_score, "reason": s.reason}
        for s in scores
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cartograph eval runner")
    p.add_argument(
        "--ids",
        nargs="+",
        metavar="ID",
        help="Run only these golden query IDs (default: all)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Ollama model name to use (default: llama3.1)",
    )
    p.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring (deterministic checks only)",
    )
    p.add_argument(
        "--min-judge-score",
        type=int,
        default=3,
        metavar="N",
        help="Minimum acceptable LLM judge score (1–5, default: 3)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    golden_set: list[dict] = json.loads(GOLDEN_SET_PATH.read_text())

    if args.ids:
        golden_set = [g for g in golden_set if g["id"] in args.ids]
        if not golden_set:
            logger.error("No golden entries matched IDs: %s", args.ids)
            return 1

    model = args.model or "llama3.1"
    run_llm_judge = not args.no_llm_judge
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"\n{'═' * 70}")
    print(f"  Cartograph Eval Run — {timestamp}")
    print(f"  Model: {model}  |  Queries: {len(golden_set)}  |  LLM judge: {run_llm_judge}")
    print(f"{'═' * 70}")

    all_results: list[dict] = []

    for golden in golden_set:
        qid = golden["id"]
        logger.info("Running pipeline for: %s", qid)

        try:
            state, elapsed = run_pipeline(golden["query"], golden["depth"], model)
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline failed for %s: %s", qid, exc)
            all_results.append({
                "id": qid,
                "golden": golden,
                "passed": False,
                "error": str(exc),
                "elapsed_seconds": 0,
            })
            continue

        det_results = deterministic.run_all(state, golden)

        judge_scores = []
        if run_llm_judge:
            judge_scores = llm_judge.run_all(state, golden, model)

        passed = _overall_pass(det_results, judge_scores, args.min_judge_score)
        _print_result(golden, state, det_results, judge_scores, elapsed)

        all_results.append({
            "id": qid,
            "golden": golden,
            "passed": passed,
            "elapsed_seconds": round(elapsed, 2),
            "pipeline_stats": _pipeline_stats(state),
            "deterministic": _serialise_det(det_results),
            "llm_judge": _serialise_judge(judge_scores),
            "sub_questions": state.get("sub_questions", []),
        })

    _print_summary(all_results)

    # Persist results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    out_path.write_text(json.dumps(
        {
            "timestamp": timestamp,
            "model": model,
            "llm_judge_enabled": run_llm_judge,
            "results": all_results,
            "summary": {
                "total": len(all_results),
                "passed": sum(1 for r in all_results if r.get("passed")),
                "failed": sum(1 for r in all_results if not r.get("passed")),
            },
        },
        indent=2,
        default=str,
    ))
    print(f"  Results saved → {out_path}\n")

    failed = sum(1 for r in all_results if not r.get("passed"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
