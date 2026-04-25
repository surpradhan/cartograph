from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.search.tavily import search


def _fake_tavily_module(client: MagicMock) -> MagicMock:
    mod = MagicMock()
    mod.TavilyClient.return_value = client
    return mod


def test_raises_if_key_empty():
    with pytest.raises(ValueError, match="Tavily API key"):
        search("test query", api_key="")


def test_raises_import_error_if_package_missing():
    with patch.dict("sys.modules", {"tavily": None}):
        with pytest.raises(ImportError, match="tavily-python"):
            search("test query", api_key="tvly-fake")


def test_returns_normalised_results():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"title": "Article A", "url": "https://example.com/a", "content": "Snippet A"},
            {"title": "Article B", "url": "https://example.com/b", "content": "Snippet B"},
        ]
    }
    with patch.dict("sys.modules", {"tavily": _fake_tavily_module(mock_client)}):
        results = search("quantum computing", max_results=2, api_key="tvly-fake")

    assert len(results) == 2
    assert results[0] == {"title": "Article A", "url": "https://example.com/a", "snippet": "Snippet A"}
    assert results[1]["snippet"] == "Snippet B"


def test_returns_empty_list_on_client_error():
    mock_client = MagicMock()
    mock_client.search.side_effect = RuntimeError("network error")
    with patch.dict("sys.modules", {"tavily": _fake_tavily_module(mock_client)}):
        results = search("test", api_key="tvly-fake")

    assert results == []


@patch("src.agent.nodes.searcher.SourceCache")
def test_searcher_dispatches_to_tavily(mock_cache_cls):
    """Searcher node picks Tavily when search_backend='tavily'."""
    from src.agent.nodes.searcher import run_searcher
    from src.config import AgentConfig

    mock_cache = MagicMock()
    mock_cache.is_duplicate.return_value = False
    mock_cache_cls.return_value = mock_cache

    cfg = AgentConfig(search_backend="tavily", tavily_api_key="tvly-fake", max_sub_questions=1)
    state = {"sub_questions": ["test question"], "search_results": []}

    mock_client = MagicMock()
    mock_client.search.return_value = {"results": [
        {"title": "T", "url": "https://x.com", "content": "S"}
    ]}
    with patch.dict("sys.modules", {"tavily": _fake_tavily_module(mock_client)}):
        output = run_searcher(state, cfg)

    assert len(output["search_results"]) == 1
    assert output["search_results"][0]["url"] == "https://x.com"


@patch("src.agent.nodes.searcher.SourceCache")
def test_searcher_dispatches_to_ddg_by_default(mock_cache_cls):
    """Searcher node uses DDG when search_backend='ddg' (the default)."""
    from src.agent.nodes.searcher import run_searcher
    from src.config import AgentConfig

    mock_cache = MagicMock()
    mock_cache.is_duplicate.return_value = False
    mock_cache_cls.return_value = mock_cache

    cfg = AgentConfig()
    state = {"sub_questions": ["test question"], "search_results": []}

    with patch("src.search.ddg.search", return_value=[
        {"title": "D", "url": "https://ddg.com", "snippet": "S"}
    ]) as mock_ddg:
        output = run_searcher(state, cfg)

    mock_ddg.assert_called_once()
    assert output["search_results"][0]["url"] == "https://ddg.com"


def test_invalid_search_backend_rejected_at_config_construction():
    """AgentConfig should reject unknown search_backend values at construction time."""
    from src.config import AgentConfig
    with pytest.raises(ValidationError):
        AgentConfig(search_backend="google")  # type: ignore[arg-type]
