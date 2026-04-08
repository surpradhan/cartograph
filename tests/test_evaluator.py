"""Tests for the Evaluator node."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import AgentConfig


@pytest.fixture
def config():
    return AgentConfig(
        model_name="llama3.1",
        min_relevance_score=3,
        min_sources_per_question=1,  # easier to satisfy in tests
        max_retries=2,
    )


def _make_source(sub_q="Q1", url="https://example.com", snippet="relevant content"):
    return {"title": "T", "url": url, "snippet": snippet, "sub_question": sub_q}


# ── Happy path ────────────────────────────────────────────────────────────────

@patch("src.agent.nodes.evaluator.ChatOllama")
def test_evaluator_scores_and_filters(mock_llm_cls, config):
    """Sources below min_relevance_score are excluded from evaluated_sources."""
    mock_llm = MagicMock()
    # First source scores 4 (pass), second scores 1 (fail)
    mock_llm.invoke.side_effect = [
        MagicMock(content='{"score": 4, "reason": "Relevant"}'),
        MagicMock(content='{"score": 1, "reason": "Off-topic"}'),
    ]
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.evaluator import run_evaluator

    state = {
        "query": "test",
        "search_results": [_make_source("Q1", "https://a.com"), _make_source("Q1", "https://b.com")],
        "sub_questions": ["Q1"],
        "retry_count": 0,
    }
    result = run_evaluator(state, config)

    assert len(result["evaluated_sources"]) == 1
    assert result["evaluated_sources"][0]["url"] == "https://a.com" or \
           result["evaluated_sources"][0]["relevance_score"] == 4


@patch("src.agent.nodes.evaluator.ChatOllama")
def test_evaluator_coverage_sufficient(mock_llm_cls, config):
    """coverage_sufficient=True when every sub-question has enough good sources."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"score": 4, "reason": "Good"}')
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.evaluator import run_evaluator

    state = {
        "query": "test",
        "search_results": [_make_source("Q1"), _make_source("Q2")],
        "sub_questions": ["Q1", "Q2"],
        "retry_count": 0,
    }
    result = run_evaluator(state, config)

    assert result["coverage_sufficient"] is True


@patch("src.agent.nodes.evaluator.ChatOllama")
def test_evaluator_coverage_insufficient_missing_sub_question(mock_llm_cls, config):
    """coverage_sufficient=False when a sub-question has no sources."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"score": 4, "reason": "Good"}')
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.evaluator import run_evaluator

    state = {
        "query": "test",
        "search_results": [_make_source("Q1")],  # Q2 has nothing
        "sub_questions": ["Q1", "Q2"],
        "retry_count": 0,
    }
    result = run_evaluator(state, config)

    assert result["coverage_sufficient"] is False


# ── Fallback / error paths ─────────────────────────────────────────────────────

@patch("src.agent.nodes.evaluator.ChatOllama")
def test_evaluator_json_parse_failure_defaults_to_score_1(mock_llm_cls, config):
    """A malformed LLM response assigns relevance_score=1 (filtered out)."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="not valid json")
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.evaluator import run_evaluator

    state = {
        "query": "test",
        "search_results": [_make_source()],
        "sub_questions": ["Q1"],
        "retry_count": 0,
    }
    result = run_evaluator(state, config)

    assert result["evaluated_sources"] == []  # score=1 < min_relevance_score=3


@patch("src.agent.nodes.evaluator.ChatOllama")
def test_evaluator_empty_search_results(mock_llm_cls, config):
    """Empty search results immediately returns coverage_sufficient=False."""
    from src.agent.nodes.evaluator import run_evaluator

    state = {
        "query": "test",
        "search_results": [],
        "sub_questions": ["Q1"],
        "retry_count": 0,
    }
    result = run_evaluator(state, config)

    assert result["coverage_sufficient"] is False
    assert result["evaluated_sources"] == []
    mock_llm_cls.assert_not_called()


@patch("src.agent.nodes.evaluator.PROMPT_PATH")
@patch("src.agent.nodes.evaluator.ChatOllama")
def test_evaluator_missing_prompt_returns_insufficient(mock_llm_cls, mock_path, config):
    """When the prompt file is missing, coverage_sufficient must be False."""
    mock_path.read_text.side_effect = OSError("file not found")

    from src.agent.nodes.evaluator import run_evaluator

    state = {
        "query": "test",
        "search_results": [_make_source()],
        "sub_questions": ["Q1"],
        "retry_count": 0,
    }
    result = run_evaluator(state, config)

    assert result["coverage_sufficient"] is False
    mock_llm_cls.assert_not_called()


@patch("src.agent.nodes.evaluator.ChatOllama")
def test_evaluator_increments_retry_count(mock_llm_cls, config):
    """retry_count in output is always input + 1."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"score": 4, "reason": "Good"}')
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.evaluator import run_evaluator

    state = {
        "query": "test",
        "search_results": [_make_source()],
        "sub_questions": ["Q1"],
        "retry_count": 1,
    }
    result = run_evaluator(state, config)

    assert result["retry_count"] == 2
