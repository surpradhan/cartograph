from unittest.mock import MagicMock, patch

import pytest

from src.config import AgentConfig
from src.llm import build_llm


def _fake_module(cls_name: str) -> MagicMock:
    mod = MagicMock()
    setattr(mod, cls_name, MagicMock())
    return mod


def test_ollama_returns_chat_ollama():
    cfg = AgentConfig(provider="ollama", model_name="llama3.1")
    mock_cls = MagicMock()
    fake_mod = _fake_module("ChatOllama")
    fake_mod.ChatOllama = mock_cls
    with patch.dict("sys.modules", {"langchain_ollama": fake_mod}):
        llm = build_llm(cfg)
    mock_cls.assert_called_once()
    assert llm is mock_cls.return_value


def test_anthropic_raises_without_key():
    cfg = AgentConfig(provider="anthropic", api_key="", model_name="claude-sonnet-4-6")
    with patch.dict("sys.modules", {"langchain_anthropic": MagicMock()}):
        with pytest.raises(ValueError, match="Anthropic API key"):
            build_llm(cfg)


def test_anthropic_raises_import_error_if_missing():
    cfg = AgentConfig(provider="anthropic", api_key="sk-ant-fake", model_name="claude-sonnet-4-6")
    with patch.dict("sys.modules", {"langchain_anthropic": None}):
        with pytest.raises(ImportError, match="langchain-anthropic"):
            build_llm(cfg)


def test_anthropic_returns_chat_anthropic():
    cfg = AgentConfig(provider="anthropic", api_key="sk-ant-fake", model_name="claude-sonnet-4-6")
    mock_cls = MagicMock()
    fake_mod = MagicMock()
    fake_mod.ChatAnthropic = mock_cls
    with patch.dict("sys.modules", {"langchain_anthropic": fake_mod}):
        llm = build_llm(cfg)
    mock_cls.assert_called_once()
    assert llm is mock_cls.return_value


def test_openai_raises_without_key():
    cfg = AgentConfig(provider="openai", api_key="", model_name="gpt-4o")
    with patch.dict("sys.modules", {"langchain_openai": MagicMock()}):
        with pytest.raises(ValueError, match="OpenAI API key"):
            build_llm(cfg)


def test_openai_raises_import_error_if_missing():
    cfg = AgentConfig(provider="openai", api_key="sk-fake", model_name="gpt-4o")
    with patch.dict("sys.modules", {"langchain_openai": None}):
        with pytest.raises(ImportError, match="langchain-openai"):
            build_llm(cfg)


def test_openai_returns_chat_openai():
    cfg = AgentConfig(provider="openai", api_key="sk-fake", model_name="gpt-4o")
    mock_cls = MagicMock()
    fake_mod = MagicMock()
    fake_mod.ChatOpenAI = mock_cls
    with patch.dict("sys.modules", {"langchain_openai": fake_mod}):
        llm = build_llm(cfg)
    mock_cls.assert_called_once()
    assert llm is mock_cls.return_value


def test_invalid_provider_rejected_at_config_construction():
    """AgentConfig should reject unknown provider values at construction time."""
    with pytest.raises(Exception):  # pydantic ValidationError
        AgentConfig(provider="gemini")  # type: ignore[arg-type]
