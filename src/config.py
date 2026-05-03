from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AgentConfig(BaseModel):
    # LLM
    provider: Literal["ollama", "anthropic", "openai"] = "ollama"
    api_key: str = ""
    model_name: str = "llama3.1"
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    llm_timeout: int = Field(default=120, gt=0)

    # Search
    results_per_query: int = Field(default=5, ge=1, le=20)
    max_sub_questions: int = Field(default=5, ge=1, le=10)

    # Evaluation
    min_relevance_score: int = Field(default=3, ge=1, le=5)
    min_sources_per_question: int = Field(default=1, ge=1)  # raise to 2+ to tighten quality gate
    max_retries: int = Field(default=2, ge=0)

    # FAISS
    dedup_threshold: float = Field(default=0.92, gt=0.0, le=1.0)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Synthesis
    snippet_max_chars: int = Field(default=500, gt=0)

    @field_validator("model_name", "embedding_model")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    model_config = {"frozen": True}
