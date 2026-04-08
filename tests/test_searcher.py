"""Tests for the Searcher node and DuckDuckGo wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import AgentConfig


@pytest.fixture
def config():
    return AgentConfig(results_per_query=3)


@patch("src.search.ddg.DDGS")
def test_ddg_search_returns_normalised_results(mock_ddgs_cls):
    """DuckDuckGo wrapper should return title/url/snippet dicts."""
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs.__exit__ = MagicMock(return_value=False)
    mock_ddgs.text.return_value = [
        {"title": "Test Title", "href": "https://example.com", "body": "Test snippet"}
    ]
    mock_ddgs_cls.return_value = mock_ddgs

    from src.search.ddg import search

    results = search("test query", max_results=1)

    assert len(results) == 1
    assert results[0]["title"] == "Test Title"
    assert results[0]["url"] == "https://example.com"
    assert results[0]["snippet"] == "Test snippet"


@patch("src.agent.nodes.searcher.SourceCache")
@patch("src.agent.nodes.searcher.search")
def test_searcher_deduplicates(mock_search, mock_cache_cls, config):
    """Searcher should skip sources flagged as duplicates by the cache."""
    from src.agent.nodes.searcher import run_searcher

    mock_search.return_value = [
        {"title": "Dup", "url": "https://dup.com", "snippet": "dup snippet"}
    ]
    mock_cache = MagicMock()
    mock_cache.is_duplicate.return_value = True  # everything is a duplicate
    mock_cache_cls.return_value = mock_cache

    state = {"sub_questions": ["Q1"]}
    result = run_searcher(state, config)

    assert result["search_results"] == []


@patch("src.agent.nodes.searcher.SourceCache")
@patch("src.agent.nodes.searcher.search")
def test_searcher_fresh_cache_per_call(mock_search, mock_cache_cls, config):
    """Each run_searcher call creates a fresh cache so retries are not blocked."""
    from src.agent.nodes.searcher import run_searcher

    mock_search.return_value = [
        {"title": "T", "url": "https://t.com", "snippet": "snippet"}
    ]
    mock_cache = MagicMock()
    mock_cache.is_duplicate.return_value = False
    mock_cache_cls.return_value = mock_cache

    state = {"sub_questions": ["Q1"]}
    run_searcher(state, config)
    run_searcher(state, config)

    # SourceCache should have been instantiated once per call
    assert mock_cache_cls.call_count == 2
