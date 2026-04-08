"""Tests for the Synthesizer node."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import AgentConfig


@pytest.fixture
def config():
    return AgentConfig(model_name="llama3.1", snippet_max_chars=500)


def _make_source(title="T", url="https://example.com", snippet="content", sub_q="Q1"):
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "sub_question": sub_q,
        "relevance_score": 4,
    }


def _base_state(sources):
    return {
        "query": "Test query",
        "sub_questions": ["Q1"],
        "evaluated_sources": sources,
    }


# ── Happy path ────────────────────────────────────────────────────────────────

@patch("src.agent.nodes.synthesizer.ChatOllama")
def test_synthesizer_returns_report_with_references(mock_llm_cls, config):
    """Report should include both LLM body and an appended References section."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="## Summary\n\nGreat topic [1].")
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.synthesizer import run_synthesizer

    result = run_synthesizer(_base_state([_make_source()]), config)

    assert "## Summary" in result["report"]
    assert "## References" in result["report"]
    assert "https://example.com" in result["report"]


@patch("src.agent.nodes.synthesizer.ChatOllama")
def test_synthesizer_strips_llm_added_references(mock_llm_cls, config):
    """LLM-generated References section is stripped before the canonical one is appended."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="## Summary\n\nText [1].\n\n## References\n\n1. Fake ref"
    )
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.synthesizer import run_synthesizer

    result = run_synthesizer(_base_state([_make_source()]), config)

    # Should contain exactly one References heading
    assert result["report"].count("## References") == 1
    # The fake ref should be gone, replaced by the canonical one
    assert "Fake ref" not in result["report"]


# ── Fallback paths ─────────────────────────────────────────────────────────────

@patch("src.agent.nodes.synthesizer.ChatOllama")
def test_synthesizer_empty_sources_returns_minimal_report(mock_llm_cls, config):
    """Empty source list returns a minimal report without calling the LLM."""
    from src.agent.nodes.synthesizer import run_synthesizer

    result = run_synthesizer(_base_state([]), config)

    assert "No sources" in result["report"]
    mock_llm_cls.assert_not_called()


@patch("src.agent.nodes.synthesizer.ChatOllama")
def test_synthesizer_llm_failure_returns_fallback_report(mock_llm_cls, config):
    """LLM exception produces a fallback report that still includes source URLs."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("ollama gone")
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.synthesizer import run_synthesizer

    result = run_synthesizer(_base_state([_make_source(url="https://fallback.com")]), config)

    assert "https://fallback.com" in result["report"]


@patch("src.agent.nodes.synthesizer.PROMPT_PATH")
@patch("src.agent.nodes.synthesizer.ChatOllama")
def test_synthesizer_missing_prompt_uses_fallback_prompt(mock_llm_cls, mock_path, config):
    """Missing prompt file uses a hardcoded fallback; LLM is still called."""
    mock_path.read_text.side_effect = OSError("missing")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Fallback report body.")
    mock_llm_cls.return_value = mock_llm

    from src.agent.nodes.synthesizer import run_synthesizer

    result = run_synthesizer(_base_state([_make_source()]), config)

    assert mock_llm.invoke.called
    assert "Fallback report body." in result["report"]
