from typing import TypedDict


class ResearchState(TypedDict):
    query: str                      # Original user query
    sub_questions: list[str]        # Decomposed sub-questions from Planner
    search_results: list[dict]      # Raw results from DuckDuckGo
    evaluated_sources: list[dict]   # Scored + filtered sources
    coverage_sufficient: bool       # Evaluator's verdict — proceed or retry?
    retry_count: int                # Track retries to prevent infinite loops
    report: str                     # Final synthesized report
