import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.agent.state import ResearchState
from src.config import AgentConfig

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "synthesizer.txt"

_REFERENCES_HEADING = "## References"


def _build_sources_block(sources: list[dict], snippet_max_chars: int) -> tuple[str, str]:
    """
    Build an inline citation map and a references section from evaluated sources.
    Returns (citation_context, references_markdown).
    """
    lines = []
    ref_lines = [f"\n\n---\n\n{_REFERENCES_HEADING}\n"]
    for i, src in enumerate(sources, start=1):
        lines.append(
            f"[{i}] Title: {src.get('title', 'Untitled')}\n"
            f"    URL: {src.get('url', '')}\n"
            f"    Sub-question: {src.get('sub_question', '')}\n"
            f"    Snippet: {src.get('snippet', '')[:snippet_max_chars]}"
        )
        ref_lines.append(f"{i}. [{src.get('title', 'Untitled')}]({src.get('url', '')})")
    return "\n".join(lines), "\n".join(ref_lines)


def run_synthesizer(state: ResearchState, config: AgentConfig) -> dict:
    """
    Generate a structured markdown research report with inline citations.

    Input state keys:  query, sub_questions, evaluated_sources
    Output state keys: report
    """
    sources = state["evaluated_sources"]
    if not sources:
        logger.warning("Synthesizer called with no evaluated sources; generating minimal report")
        return {
            "report": (
                f"# Research Report: {state['query']}\n\n"
                "_No sources with sufficient relevance were found. "
                "Try rephrasing your query or increasing the search depth._"
            )
        }

    try:
        system_prompt = PROMPT_PATH.read_text()
    except OSError as exc:
        logger.error("Could not read synthesizer prompt from %s: %s", PROMPT_PATH, exc)
        system_prompt = (
            "You are a research assistant. Write a clear, well-structured markdown report "
            "that answers the research query using the provided sources. "
            "Use [N] inline citations. "
            "Do NOT include a References or Sources section at the end."
        )

    llm = ChatOllama(model=config.model_name, temperature=config.temperature, timeout=config.llm_timeout)
    citation_context, references = _build_sources_block(sources, config.snippet_max_chars)

    user_message = (
        f"Research query: {state['query']}\n\n"
        f"Sub-questions to address:\n"
        + "\n".join(f"- {q}" for q in state["sub_questions"])
        + f"\n\nSources (use [N] for inline citations):\n{citation_context}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        body = response.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Synthesizer LLM call failed: %s", exc)
        source_lines = "\n".join(
            f"- [{s.get('title', 'Untitled')}]({s.get('url', '')}): {s.get('snippet', '')[:200]}"
            for s in sources
        )
        report = (
            f"# Research Report: {state['query']}\n\n"
            f"_Report generation failed ({exc}). Raw sources below._\n\n"
            f"{source_lines}"
            + references
        )
        logger.info("Synthesizer produced fallback report of %d characters", len(report))
        return {"report": report}

    # Guard against the LLM adding its own references section despite instructions
    if _REFERENCES_HEADING in body:
        body = body[:body.index(_REFERENCES_HEADING)].rstrip()
        logger.warning("LLM included a References section despite instructions; stripped it")

    report = body + references
    logger.info("Synthesizer produced a report of %d characters", len(report))
    return {"report": report}
