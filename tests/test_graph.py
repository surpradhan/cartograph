"""Integration-style tests for the LangGraph state machine."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import AgentConfig


@pytest.fixture
def config():
    return AgentConfig(
        model_name="llama3.1",
        max_sub_questions=3,
        max_retries=1,
        min_sources_per_question=1,
    )


def _make_llm_mock(content: str) -> MagicMock:
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=content)
    return mock


def _initial_state():
    return {
        "query": "Test research topic",
        "sub_questions": [],
        "search_results": [],
        "evaluated_sources": [],
        "coverage_sufficient": False,
        "retry_count": 0,
        "report": "",
    }


def _patch_cache():
    """Patch SourceCache so tests don't load sentence-transformers."""
    mock_cache = MagicMock()
    mock_cache.is_duplicate.return_value = False
    mock_cache.add.return_value = None
    return patch("src.agent.nodes.searcher.SourceCache", return_value=mock_cache)


# ── Happy path ────────────────────────────────────────────────────────────────

@patch("src.agent.nodes.synthesizer.ChatOllama")
@patch("src.agent.nodes.evaluator.ChatOllama")
@patch("src.agent.nodes.planner.ChatOllama")
@patch("src.agent.nodes.searcher.search")
def test_graph_happy_path(
    mock_search,
    mock_planner_llm_cls,
    mock_evaluator_llm_cls,
    mock_synthesizer_llm_cls,
    config,
):
    """Graph should run end-to-end and return a non-empty report."""
    with _patch_cache():
        mock_search.return_value = [
            {"title": "S1", "url": "https://s1.com", "snippet": "Relevant.", "sub_question": "Q1"},
        ]
        mock_planner_llm_cls.return_value = _make_llm_mock("1. Q1\n2. Q2\n3. Q3")
        mock_evaluator_llm_cls.return_value = _make_llm_mock('{"score": 4, "reason": "Relevant"}')
        mock_synthesizer_llm_cls.return_value = _make_llm_mock("## Summary\n\nReport [1].")

        from src.agent.graph import build_graph

        result = build_graph(config).invoke(_initial_state())

    assert result["report"]
    assert "## Summary" in result["report"]


# ── Unhappy paths ─────────────────────────────────────────────────────────────

@patch("src.agent.nodes.synthesizer.ChatOllama")
@patch("src.agent.nodes.evaluator.ChatOllama")
@patch("src.agent.nodes.planner.ChatOllama")
@patch("src.agent.nodes.searcher.search")
def test_graph_planner_fallback_to_original_query(
    mock_search,
    mock_planner_llm_cls,
    mock_evaluator_llm_cls,
    mock_synthesizer_llm_cls,
    config,
):
    """When planner LLM fails, the original query becomes the single sub-question."""
    with _patch_cache():
        mock_planner_llm_cls.return_value.invoke.side_effect = RuntimeError("ollama down")
        mock_search.return_value = [
            {"title": "T", "url": "https://t.com", "snippet": "content", "sub_question": "Test research topic"},
        ]
        mock_evaluator_llm_cls.return_value = _make_llm_mock('{"score": 4, "reason": "OK"}')
        mock_synthesizer_llm_cls.return_value = _make_llm_mock("Fallback report.")

        from src.agent.graph import build_graph

        result = build_graph(config).invoke(_initial_state())

    assert result["sub_questions"] == ["Test research topic"]
    assert result["report"]


@patch("src.agent.nodes.synthesizer.ChatOllama")
@patch("src.agent.nodes.evaluator.ChatOllama")
@patch("src.agent.nodes.planner.ChatOllama")
@patch("src.agent.nodes.searcher.search")
def test_graph_all_searches_fail_produces_minimal_report(
    mock_search,
    mock_planner_llm_cls,
    mock_evaluator_llm_cls,
    mock_synthesizer_llm_cls,
    config,
):
    """When all searches return empty, the graph eventually reaches synthesizer with no sources."""
    with _patch_cache():
        mock_search.return_value = []  # all searches fail
        mock_planner_llm_cls.return_value = _make_llm_mock("1. Q1")
        mock_evaluator_llm_cls.return_value = _make_llm_mock('{"score": 4, "reason": "OK"}')
        mock_synthesizer_llm_cls.return_value = _make_llm_mock("No sources report.")

        from src.agent.graph import build_graph

        result = build_graph(config).invoke(_initial_state())

    # Either synthesizer returned a proper report or the empty-sources fallback
    assert result["report"]


@patch("src.agent.nodes.synthesizer.ChatOllama")
@patch("src.agent.nodes.evaluator.ChatOllama")
@patch("src.agent.nodes.planner.ChatOllama")
@patch("src.agent.nodes.searcher.search")
def test_graph_max_retries_exhausted_reaches_synthesizer(
    mock_search,
    mock_planner_llm_cls,
    mock_evaluator_llm_cls,
    mock_synthesizer_llm_cls,
    config,
):
    """Graph reaches synthesizer after max_retries even if coverage is never sufficient."""
    with _patch_cache():
        mock_search.return_value = [
            {"title": "T", "url": "https://t.com", "snippet": "content", "sub_question": "Q1"},
        ]
        mock_planner_llm_cls.return_value = _make_llm_mock("1. Q1\n2. Q2")  # Q2 never covered
        # Always returns low score so coverage stays insufficient
        mock_evaluator_llm_cls.return_value = _make_llm_mock('{"score": 1, "reason": "bad"}')
        mock_synthesizer_llm_cls.return_value = _make_llm_mock("Partial report.")

        from src.agent.graph import build_graph

        result = build_graph(config).invoke(_initial_state())

    # Should still produce a report (possibly empty-sources fallback)
    assert result["report"]
    # retry_count should have exceeded max_retries
    assert result["retry_count"] > config.max_retries
