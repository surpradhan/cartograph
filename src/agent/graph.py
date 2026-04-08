from langgraph.graph import END, StateGraph

from src.agent.nodes.evaluator import run_evaluator
from src.agent.nodes.planner import run_planner
from src.agent.nodes.searcher import run_searcher
from src.agent.nodes.synthesizer import run_synthesizer
from src.agent.state import ResearchState
from src.config import AgentConfig


def _should_retry(config: AgentConfig):
    """
    Return a conditional-edge function that respects config.max_retries.

    retry_count is incremented each time the evaluator runs.  We stop retrying
    when retry_count *exceeds* max_retries (i.e. > not >=), which gives the
    user the expected number of actual retry cycles.
    """
    def _check(state: ResearchState) -> str:
        if state["coverage_sufficient"] or state["retry_count"] > config.max_retries:
            return "synthesizer"
        return "searcher"
    return _check


def build_graph(config: AgentConfig) -> StateGraph:
    """Build and compile the Cartograph LangGraph state machine."""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", lambda s: run_planner(s, config))
    graph.add_node("searcher", lambda s: run_searcher(s, config))
    graph.add_node("evaluator", lambda s: run_evaluator(s, config))
    graph.add_node("synthesizer", lambda s: run_synthesizer(s, config))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _should_retry(config),
        {"searcher": "searcher", "synthesizer": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    return graph.compile()
