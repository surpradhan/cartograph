import logging

from src.agent.state import ResearchState
from src.config import AgentConfig
from src.retrieval.cache import SourceCache
from src.search.ddg import search

logger = logging.getLogger(__name__)


def run_searcher(state: ResearchState, config: AgentConfig) -> dict:
    """
    Run sequential DuckDuckGo searches for each sub-question, deduplicating
    results via a fresh FAISS source cache.

    A new cache is created on every call so that retry cycles are not blocked
    by results that were already found (and cached) in a previous cycle.

    Note: ddgs uses primp (a Rust HTTP library) which is not thread-safe.
    Searches run sequentially to avoid hangs in ThreadPoolExecutor.

    Input state keys:  sub_questions
    Output state keys: search_results
    """
    cache = SourceCache(
        model_name=config.embedding_model,
        threshold=config.dedup_threshold,
    )
    sub_questions = state["sub_questions"]
    if not sub_questions:
        logger.warning("Searcher called with no sub-questions; returning empty results")
        return {"search_results": []}

    all_results: list[dict] = []

    for q in sub_questions:
        try:
            results = search(q, max_results=config.results_per_query)
            if not results:
                logger.warning("No results returned for sub-question: '%s'", q)
                continue
            added = 0
            for result in results:
                result["sub_question"] = q
                snippet = result.get("snippet", "")
                url = result.get("url", "")
                if not cache.is_duplicate(snippet, url):
                    cache.add(snippet, url)
                    all_results.append(result)
                    added += 1
            logger.info("Sub-question '%s': %d new results (after dedup)", q, added)
        except Exception as exc:  # noqa: BLE001
            logger.error("Search failed for '%s': %s", q, exc)

    logger.info("Searcher collected %d total unique results", len(all_results))
    return {"search_results": all_results}
