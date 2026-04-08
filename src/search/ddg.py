import logging
import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds


def search(query: str, max_results: int = 5) -> list[dict]:
    """
    Run a DuckDuckGo text search and return normalised results.

    Note: ddgs uses primp (Rust) which is not thread-safe — always call
    from the main thread or a single-threaded context.

    Retries up to _MAX_RETRIES times with exponential backoff.
    Returns an empty list if all attempts fail.
    """
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
        except RatelimitException as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** (attempt + 1)
            logger.warning("DDG rate-limited (attempt %d/%d) for '%s' — waiting %.1fs",
                           attempt + 1, _MAX_RETRIES, query, wait)
            time.sleep(wait)
        except (DDGSException, Exception) as exc:  # noqa: BLE001
            last_exc = exc
            wait = _BACKOFF_BASE ** attempt
            logger.warning("Search error (attempt %d/%d) for '%s': %s — retrying in %.1fs",
                           attempt + 1, _MAX_RETRIES, query, exc, wait)
            time.sleep(wait)

    logger.error("All %d search attempts failed for '%s': %s", _MAX_RETRIES, query, last_exc)
    return []
