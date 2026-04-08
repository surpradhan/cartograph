from dataclasses import dataclass


@dataclass
class AgentConfig:
    # LLM
    model_name: str = "llama3.1"
    temperature: float = 0.3
    llm_timeout: int = 120  # seconds; applies to all Ollama calls

    # Search
    results_per_query: int = 5
    max_sub_questions: int = 5

    # Evaluation
    min_relevance_score: int = 3
    min_sources_per_question: int = 2
    max_retries: int = 2

    # FAISS
    dedup_threshold: float = 0.92
    embedding_model: str = "all-MiniLM-L6-v2"

    # Synthesis
    snippet_max_chars: int = 500
