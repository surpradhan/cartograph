import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.agent.state import ResearchState
from src.config import AgentConfig

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner.txt"


def run_planner(state: ResearchState, config: AgentConfig) -> dict:
    """
    Decompose the user's query into 3-5 focused sub-questions.

    Input state keys:  query
    Output state keys: sub_questions
    """
    try:
        system_prompt = PROMPT_PATH.read_text()
    except OSError as exc:
        logger.error("Could not read planner prompt from %s: %s", PROMPT_PATH, exc)
        # Fall back to the query itself as the sole sub-question
        return {"sub_questions": [state["query"]]}

    llm = ChatOllama(model=config.model_name, temperature=config.temperature, timeout=config.llm_timeout)

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Query: {state['query']}\nMax sub-questions: {config.max_sub_questions}"
            ),
        ])
        raw = response.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Planner LLM call failed: %s", exc)
        return {"sub_questions": [state["query"]]}

    sub_questions = [
        line.lstrip("0123456789.-) ").strip()
        for line in raw.splitlines()
        if line.strip()
    ][:config.max_sub_questions]

    if not sub_questions:
        logger.warning("Planner returned an empty response; using original query as fallback")
        sub_questions = [state["query"]]

    logger.info("Planner produced %d sub-questions", len(sub_questions))
    return {"sub_questions": sub_questions}
