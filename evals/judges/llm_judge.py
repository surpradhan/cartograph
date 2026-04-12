"""
LLM-as-judge eval checks.
Uses Ollama locally — requires the configured model to be running.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class JudgeScore:
    dimension: str
    score: int        # 1–5
    max_score: int = 5
    reason: str = ""
    raw: str = field(default="", repr=False)  # raw LLM output for debugging


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_ollama(model: str, prompt: str, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "")


def _parse_score(text: str) -> tuple[int, str]:
    """Extract (score, reason) from LLM response. Tries JSON first, then regex."""
    # JSON extraction
    try:
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = int(data.get("score", 0))
            reason = str(data.get("reason", "")).strip()
            if 1 <= score <= 5:
                return score, reason
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Fallback: plain number patterns
    for pattern in [r"score[:\s]+([1-5])", r"([1-5])\s*/\s*5", r"\b([1-5])\b"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1)), text[:300].strip()

    return 1, f"Could not parse score from response: {text[:200]}"


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_sub_question_quality(
    query: str,
    sub_questions: list[str],
    model: str,
) -> JudgeScore:
    """Score relevance + diversity of the planner's sub-questions."""
    sqs = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_questions))
    prompt = f"""You are evaluating a research agent's question decomposition.

Original query: "{query}"

Generated sub-questions:
{sqs}

Score on TWO criteria combined:
1. Relevance — do all sub-questions relate directly to the original query?
2. Diversity — do they cover meaningfully different aspects (not just rephrasings)?

Respond with JSON only, no extra text:
{{"score": <1-5>, "reason": "<one concise sentence>"}}

Scoring guide:
5 = all relevant AND cover clearly distinct aspects
3 = mostly relevant but some overlap or minor drift
1 = mostly irrelevant or all nearly identical"""

    raw = _call_ollama(model, prompt)
    score, reason = _parse_score(raw)
    return JudgeScore(
        dimension="sub_question_quality",
        score=score,
        reason=reason,
        raw=raw,
    )


def score_report_coherence(
    query: str,
    report: str,
    model: str,
) -> JudgeScore:
    """Score logical flow, structure, and how well the report answers the query."""
    truncated = report[:3000] + ("…" if len(report) > 3000 else "")
    prompt = f"""You are evaluating a research report for coherence and completeness.

Original query: "{query}"

Report (may be truncated):
{truncated}

Score on THREE criteria combined:
1. Structure — does it have clear sections and logical flow?
2. Completeness — does it meaningfully answer the original query?
3. Readability — is the writing clear and non-repetitive?

Respond with JSON only, no extra text:
{{"score": <1-5>, "reason": "<one concise sentence>"}}

Scoring guide:
5 = well-structured, fully answers the query, clear prose
3 = answers the query but has structural or clarity issues
1 = incoherent, off-topic, or mostly empty"""

    raw = _call_ollama(model, prompt)
    score, reason = _parse_score(raw)
    return JudgeScore(
        dimension="report_coherence",
        score=score,
        reason=reason,
        raw=raw,
    )


def score_factual_grounding(
    report: str,
    evaluated_sources: list[dict],
    model: str,
) -> JudgeScore:
    """Score how well the report grounds claims in cited sources."""
    source_titles = "\n".join(
        f"- {s.get('title', 'Unknown')}" for s in evaluated_sources[:10]
    )
    truncated = report[:2500] + ("…" if len(report) > 2500 else "")
    prompt = f"""You are evaluating whether a research report is grounded in its sources.

Sources available to the agent:
{source_titles}

Report excerpt:
{truncated}

Score on TWO criteria combined:
1. Citation usage — does the report use inline citations like [1], [2]?
2. Consistency — do the claims seem consistent with what the source titles suggest?

Respond with JSON only, no extra text:
{{"score": <1-5>, "reason": "<one concise sentence>"}}

Scoring guide:
5 = citations throughout, claims well-supported
3 = some citations, a few unsupported claims
1 = no citations or claims contradict the sources"""

    raw = _call_ollama(model, prompt)
    score, reason = _parse_score(raw)
    return JudgeScore(
        dimension="factual_grounding",
        score=score,
        reason=reason,
        raw=raw,
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all(
    state: dict,
    golden: dict,
    model: str,
) -> list[JudgeScore]:
    """Run all LLM judge dimensions. Failures are logged and skipped."""
    scores: list[JudgeScore] = []
    query = golden["query"]

    dimensions = [
        (
            "sub_question_quality",
            lambda: score_sub_question_quality(
                query, state.get("sub_questions", []), model
            ),
        ),
        (
            "report_coherence",
            lambda: score_report_coherence(
                query, state.get("report", ""), model
            ),
        ),
        (
            "factual_grounding",
            lambda: score_factual_grounding(
                state.get("report", ""),
                state.get("evaluated_sources", []),
                model,
            ),
        ),
    ]

    for name, fn in dimensions:
        try:
            scores.append(fn())
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM judge '%s' failed: %s", name, exc)
            scores.append(JudgeScore(
                dimension=name,
                score=0,
                reason=f"Judge failed: {exc}",
            ))

    return scores
