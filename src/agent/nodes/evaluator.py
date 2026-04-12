import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.agent.state import ResearchState
from src.config import AgentConfig

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evaluator.txt"


def _score_source(llm: ChatOllama, source: dict, query: str, system_prompt: str) -> dict:
    """Ask the LLM to score a single source on a 1-5 relevance scale."""
    content = (
        f"Research query: {query}\n"
        f"Sub-question: {source.get('sub_question', '')}\n"
        f"Title: {source.get('title', '')}\n"
        f"Snippet: {source.get('snippet', '')}\n\n"
        "Respond with a JSON object: {\"score\": <1-5>, \"reason\": \"<one sentence>\"}"
    )
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ])
        data = json.loads(response.content.strip())
        source["relevance_score"] = int(data.get("score", 1))
        source["score_reason"] = data.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        logger.warning("Could not parse LLM JSON for source '%s'", source.get("url", ""))
        source["relevance_score"] = 1
        source["score_reason"] = "Could not parse LLM response"
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM scoring call failed for source '%s': %s", source.get("url", ""), exc)
        source["relevance_score"] = 1
        source["score_reason"] = f"LLM error: {exc}"
    return source


def run_evaluator(state: ResearchState, config: AgentConfig) -> dict:
    """
    Score each source for relevance, filter below threshold, and decide
    whether coverage is sufficient to proceed to synthesis.

    On retries, URLs already present in evaluated_sources are skipped to avoid
    re-scoring the same content. New passing sources are merged into the
    accumulated evaluated_sources before the coverage check runs.

    Input state keys:  search_results, sub_questions, retry_count, evaluated_sources
    Output state keys: evaluated_sources, coverage_sufficient, retry_count
    """
    retry_count = state.get("retry_count", 0) + 1

    # Deduplicate across retries: skip URLs already present in evaluated_sources
    already_evaluated_urls = {src.get("url") for src in state.get("evaluated_sources", [])}
    search_results = [
        s for s in state["search_results"]
        if s.get("url") not in already_evaluated_urls
    ]
    if already_evaluated_urls and len(search_results) < len(state["search_results"]):
        logger.info(
            "Evaluator: skipped %d already-evaluated URL(s) from prior retries",
            len(state["search_results"]) - len(search_results),
        )

    if not search_results:
        logger.warning(
            "Evaluator received no new search results; carrying forward prior evaluated sources"
        )
        return {
            "evaluated_sources": state.get("evaluated_sources", []),
            "coverage_sufficient": False,
            "retry_count": retry_count,
        }

    try:
        system_prompt = PROMPT_PATH.read_text()
    except OSError as exc:
        logger.error("Could not read evaluator prompt from %s: %s", PROMPT_PATH, exc)
        # Cannot score without a prompt — mark insufficient so the pipeline retries
        return {
            "evaluated_sources": [],
            "coverage_sufficient": False,
            "retry_count": retry_count,
        }

    llm = ChatOllama(model=config.model_name, temperature=0.0, timeout=config.llm_timeout)
    query = state["query"]

    # Score all sources in parallel (Ollama handles one request at a time, but parallelism
    # ensures we don't bottleneck on Python between calls)
    max_workers = min(len(search_results), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_score_source, llm, dict(src), query, system_prompt)
            for src in search_results
        ]
        scored = [f.result() for f in as_completed(futures)]

    # Keep sources that meet the minimum relevance threshold
    evaluated = [s for s in scored if s["relevance_score"] >= config.min_relevance_score]
    logger.info(
        "Evaluator: %d/%d sources passed relevance threshold %d",
        len(evaluated), len(scored), config.min_relevance_score,
    )

    # Carry forward sources from previous retries so coverage compounds across retries
    all_evaluated = state.get("evaluated_sources", []) + evaluated

    # Re-run coverage check against the full accumulated set
    sub_q_counts = Counter(src.get("sub_question") for src in all_evaluated)
    coverage_sufficient = all(
        sub_q_counts.get(q, 0) >= config.min_sources_per_question
        for q in state["sub_questions"]
    )
    covered = sum(
        1 for q in state["sub_questions"]
        if sub_q_counts.get(q, 0) >= config.min_sources_per_question
    )
    logger.info(
        "Coverage (cumulative): %d/%d sub-questions have >=%d sources — sufficient=%s",
        covered, len(state["sub_questions"]), config.min_sources_per_question, coverage_sufficient,
    )

    return {
        "evaluated_sources": all_evaluated,
        "coverage_sufficient": coverage_sufficient,
        "retry_count": retry_count,
    }
