import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import AgentConfig

logger = logging.getLogger(__name__)

_ANTHROPIC_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

_OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
]

CLOUD_MODEL_CHOICES = {
    "Anthropic": _ANTHROPIC_MODELS,
    "OpenAI": _OPENAI_MODELS,
}


def build_llm(config: "AgentConfig"):
    """
    Return a LangChain chat model for the configured provider.

    Providers:
      - "ollama"    → ChatOllama (local, no key required)
      - "anthropic" → ChatAnthropic (requires api_key; install langchain-anthropic)
      - "openai"    → ChatOpenAI (requires api_key; install langchain-openai)
    """
    if config.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is not installed. Run: uv add langchain-anthropic"
            ) from exc
        if not config.api_key:
            raise ValueError("Anthropic API key is required. Enter it in the UI.")
        return ChatAnthropic(
            model=config.model_name,
            api_key=config.api_key,
            timeout=config.llm_timeout,
            temperature=config.temperature,
        )

    if config.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is not installed. Run: uv add langchain-openai"
            ) from exc
        if not config.api_key:
            raise ValueError("OpenAI API key is required. Enter it in the UI.")
        return ChatOpenAI(
            model=config.model_name,
            api_key=config.api_key,
            timeout=config.llm_timeout,
            temperature=config.temperature,
        )

    # Default: Ollama
    from langchain_ollama import ChatOllama  # noqa: PLC0415
    return ChatOllama(
        model=config.model_name,
        temperature=config.temperature,
        timeout=config.llm_timeout,
    )
