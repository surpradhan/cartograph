import pytest
from pydantic import ValidationError

from src.config import AgentConfig


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_defaults_are_valid():
    cfg = AgentConfig()
    assert cfg.model_name == "llama3.1"
    assert cfg.temperature == 0.3
    assert cfg.llm_timeout == 120
    assert cfg.results_per_query == 5
    assert cfg.max_sub_questions == 5
    assert cfg.min_relevance_score == 3
    assert cfg.min_sources_per_question == 1
    assert cfg.max_retries == 2
    assert cfg.dedup_threshold == 0.92
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.snippet_max_chars == 500


# ---------------------------------------------------------------------------
# Valid edge cases
# ---------------------------------------------------------------------------

def test_temperature_boundary_values():
    assert AgentConfig(temperature=0.0).temperature == 0.0
    assert AgentConfig(temperature=1.0).temperature == 1.0


def test_max_retries_zero_is_valid():
    assert AgentConfig(max_retries=0).max_retries == 0


def test_dedup_threshold_boundary():
    assert AgentConfig(dedup_threshold=1.0).dedup_threshold == 1.0
    assert AgentConfig(dedup_threshold=0.01).dedup_threshold == 0.01


def test_results_per_query_boundaries():
    assert AgentConfig(results_per_query=1).results_per_query == 1
    assert AgentConfig(results_per_query=20).results_per_query == 20


def test_max_sub_questions_boundaries():
    assert AgentConfig(max_sub_questions=1).max_sub_questions == 1
    assert AgentConfig(max_sub_questions=10).max_sub_questions == 10


def test_min_relevance_score_boundaries():
    assert AgentConfig(min_relevance_score=1).min_relevance_score == 1
    assert AgentConfig(min_relevance_score=5).min_relevance_score == 5


# ---------------------------------------------------------------------------
# Frozen — instances must be immutable
# ---------------------------------------------------------------------------

def test_frozen_prevents_mutation():
    cfg = AgentConfig()
    with pytest.raises((TypeError, ValidationError)):
        cfg.model_name = "something-else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Invalid — temperature
# ---------------------------------------------------------------------------

def test_temperature_below_zero_raises():
    with pytest.raises(ValidationError, match="temperature"):
        AgentConfig(temperature=-0.1)


def test_temperature_above_one_raises():
    with pytest.raises(ValidationError, match="temperature"):
        AgentConfig(temperature=1.1)


# ---------------------------------------------------------------------------
# Invalid — llm_timeout
# ---------------------------------------------------------------------------

def test_llm_timeout_zero_raises():
    with pytest.raises(ValidationError, match="llm_timeout"):
        AgentConfig(llm_timeout=0)


def test_llm_timeout_negative_raises():
    with pytest.raises(ValidationError, match="llm_timeout"):
        AgentConfig(llm_timeout=-10)


# ---------------------------------------------------------------------------
# Invalid — results_per_query
# ---------------------------------------------------------------------------

def test_results_per_query_zero_raises():
    with pytest.raises(ValidationError, match="results_per_query"):
        AgentConfig(results_per_query=0)


def test_results_per_query_too_high_raises():
    with pytest.raises(ValidationError, match="results_per_query"):
        AgentConfig(results_per_query=21)


# ---------------------------------------------------------------------------
# Invalid — max_sub_questions
# ---------------------------------------------------------------------------

def test_max_sub_questions_zero_raises():
    with pytest.raises(ValidationError, match="max_sub_questions"):
        AgentConfig(max_sub_questions=0)


def test_max_sub_questions_too_high_raises():
    with pytest.raises(ValidationError, match="max_sub_questions"):
        AgentConfig(max_sub_questions=11)


# ---------------------------------------------------------------------------
# Invalid — min_relevance_score
# ---------------------------------------------------------------------------

def test_min_relevance_score_zero_raises():
    with pytest.raises(ValidationError, match="min_relevance_score"):
        AgentConfig(min_relevance_score=0)


def test_min_relevance_score_six_raises():
    with pytest.raises(ValidationError, match="min_relevance_score"):
        AgentConfig(min_relevance_score=6)


# ---------------------------------------------------------------------------
# Invalid — max_retries
# ---------------------------------------------------------------------------

def test_max_retries_negative_raises():
    with pytest.raises(ValidationError, match="max_retries"):
        AgentConfig(max_retries=-1)


# ---------------------------------------------------------------------------
# Invalid — dedup_threshold
# ---------------------------------------------------------------------------

def test_dedup_threshold_zero_raises():
    with pytest.raises(ValidationError, match="dedup_threshold"):
        AgentConfig(dedup_threshold=0.0)


def test_dedup_threshold_above_one_raises():
    with pytest.raises(ValidationError, match="dedup_threshold"):
        AgentConfig(dedup_threshold=1.01)


# ---------------------------------------------------------------------------
# Invalid — snippet_max_chars
# ---------------------------------------------------------------------------

def test_snippet_max_chars_zero_raises():
    with pytest.raises(ValidationError, match="snippet_max_chars"):
        AgentConfig(snippet_max_chars=0)


def test_snippet_max_chars_negative_raises():
    with pytest.raises(ValidationError, match="snippet_max_chars"):
        AgentConfig(snippet_max_chars=-100)


# ---------------------------------------------------------------------------
# Invalid — blank strings
# ---------------------------------------------------------------------------

def test_model_name_blank_raises():
    with pytest.raises(ValidationError, match="model_name"):
        AgentConfig(model_name="   ")


def test_model_name_empty_raises():
    with pytest.raises(ValidationError, match="model_name"):
        AgentConfig(model_name="")


def test_embedding_model_blank_raises():
    with pytest.raises(ValidationError, match="embedding_model"):
        AgentConfig(embedding_model="")


# ---------------------------------------------------------------------------
# Invalid — wrong types
# ---------------------------------------------------------------------------

def test_temperature_wrong_type_raises():
    with pytest.raises(ValidationError):
        AgentConfig(temperature="hot")  # type: ignore[arg-type]


def test_max_retries_wrong_type_raises():
    with pytest.raises(ValidationError):
        AgentConfig(max_retries="two")  # type: ignore[arg-type]
