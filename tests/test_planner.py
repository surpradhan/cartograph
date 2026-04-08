"""Tests for the Planner node."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import AgentConfig


@pytest.fixture
def config():
    return AgentConfig(model_name="llama3.1", max_sub_questions=5)


@patch("src.agent.nodes.planner.ChatOllama")
def test_planner_returns_sub_questions(mock_llm_cls, config):
    """Planner should return a non-empty list of sub-questions."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=(
            "1. What is speculative decoding?\n"
            "2. Which models run on mobile?\n"
            "3. What is quantization?"
        )
    )
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.planner import run_planner

    state = {"query": "Latest advances in on-device LLM inference"}
    result = run_planner(state, config)

    assert "sub_questions" in result
    assert len(result["sub_questions"]) == 3
    assert "What is speculative decoding?" in result["sub_questions"]


@patch("src.agent.nodes.planner.ChatOllama")
def test_planner_respects_max_sub_questions(mock_llm_cls, config):
    """Planner should not exceed max_sub_questions."""
    config.max_sub_questions = 2
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="1. Q1\n2. Q2\n3. Q3\n4. Q4\n5. Q5"
    )
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.planner import run_planner

    state = {"query": "Test query"}
    result = run_planner(state, config)

    assert len(result["sub_questions"]) <= 2
