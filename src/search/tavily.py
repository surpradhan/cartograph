import logging

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = 5, api_key: str = "") -> list[dict]:
    """
    Run a Tavily search and return normalised results.

    Requires the `tavily-python` package: uv add tavily-python
    Raises ImportError if the package is not installed.
    Raises ValueError if api_key is empty.
    """
    if not api_key:
        raise ValueError(
            "Tavily API key is required. Get one at https://tavily.com and enter it in the UI."
        )

    try:
        from tavily import TavilyClient  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "tavily-python is not installed. Run: uv add tavily-python"
        ) from exc

    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001
        logger.error("Tavily search failed for '%s': %s", query, exc)
        return []

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]
