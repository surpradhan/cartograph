"""
Deterministic (rule-based) eval checks.
No LLM required — fast and fully reproducible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error" | "warning" | "info"


# ── Individual checks ─────────────────────────────────────────────────────────

def check_sub_question_count(
    sub_questions: list[str],
    expected_range: tuple[int, int],
) -> CheckResult:
    n = len(sub_questions)
    lo, hi = expected_range
    passed = lo <= n <= hi
    return CheckResult(
        name="sub_question_count",
        passed=passed,
        message=f"Got {n} sub-questions (expected {lo}–{hi})",
        severity="error",
    )


def check_sub_questions_non_empty(sub_questions: list[str]) -> CheckResult:
    empties = [i for i, q in enumerate(sub_questions) if not q.strip()]
    passed = len(empties) == 0
    return CheckResult(
        name="sub_questions_non_empty",
        passed=passed,
        message=(
            "All sub-questions are non-empty"
            if passed
            else f"Empty sub-questions at indices: {empties}"
        ),
        severity="error",
    )


def check_source_count(
    evaluated_sources: list[dict],
    min_sources: int,
) -> CheckResult:
    n = len(evaluated_sources)
    passed = n >= min_sources
    return CheckResult(
        name="source_count",
        passed=passed,
        message=f"Got {n} evaluated sources (minimum {min_sources})",
        severity="error",
    )


def check_no_duplicate_urls(evaluated_sources: list[dict]) -> CheckResult:
    urls = [s.get("url", "") for s in evaluated_sources]
    dupes = [u for u in set(urls) if urls.count(u) > 1]
    passed = len(dupes) == 0
    return CheckResult(
        name="no_duplicate_urls",
        passed=passed,
        message=(
            "No duplicate source URLs"
            if passed
            else f"Duplicate URLs: {dupes[:3]}"
        ),
        severity="warning",
    )


def check_report_has_references(report: str) -> CheckResult:
    passed = bool(re.search(r"##\s*References", report, re.IGNORECASE))
    return CheckResult(
        name="report_has_references",
        passed=passed,
        message=(
            "Report contains References section"
            if passed
            else "Report is missing References section"
        ),
        severity="error",
    )


def check_report_has_headings(report: str) -> CheckResult:
    headings = re.findall(r"^#{1,4}\s+.+", report, re.MULTILINE)
    passed = len(headings) >= 2
    return CheckResult(
        name="report_has_headings",
        passed=passed,
        message=f"Report has {len(headings)} heading(s)",
        severity="warning",
    )


def check_report_word_count(
    report: str,
    min_words: int,
    max_words: int,
) -> CheckResult:
    words = len(report.split())
    passed = min_words <= words <= max_words
    return CheckResult(
        name="report_word_count",
        passed=passed,
        message=f"Report has {words} words (expected {min_words}–{max_words})",
        severity="warning",
    )


def check_report_has_citations(report: str) -> CheckResult:
    citations = re.findall(r"\[\d+\]", report)
    passed = len(citations) >= 2
    return CheckResult(
        name="report_has_citations",
        passed=passed,
        message=f"Report has {len(citations)} inline citation(s)",
        severity="warning",
    )


def check_required_topics(
    report: str,
    required_topics: list[str | list[str]],
) -> list[CheckResult]:
    """Check that required topics appear in the report.

    Each entry can be a string (exact substring match) or a list of strings
    (any one match counts — use for synonyms, e.g. ["civilization", "societies"]).
    """
    results = []
    report_lower = report.lower()
    for topic in required_topics:
        if isinstance(topic, list):
            aliases = topic
            label = "/".join(aliases)
            found = any(alias.lower() in report_lower for alias in aliases)
        else:
            aliases = [topic]
            label = topic
            found = topic.lower() in report_lower
        results.append(CheckResult(
            name=f"topic_coverage:{label}",
            passed=found,
            message=(
                f"Required topic '{label}' found in report"
                if found
                else f"Required topic '{label}' NOT found in report"
            ),
            severity="error" if not found else "info",
        ))
    return results


def check_forbidden_topics(
    report: str,
    forbidden_topics: list[str],
) -> list[CheckResult]:
    results = []
    report_lower = report.lower()
    for topic in forbidden_topics:
        found = topic.lower() in report_lower
        results.append(CheckResult(
            name=f"forbidden_topic:{topic}",
            passed=not found,
            message=(
                f"Forbidden topic '{topic}' not present"
                if not found
                else f"Forbidden topic '{topic}' FOUND in report"
            ),
            severity="error" if found else "info",
        ))
    return results


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all(
    state: dict[str, Any],
    golden: dict[str, Any],
) -> list[CheckResult]:
    """Run all deterministic checks against a pipeline result and golden entry."""
    results: list[CheckResult] = []

    sub_questions: list[str] = state.get("sub_questions", [])
    evaluated_sources: list[dict] = state.get("evaluated_sources", [])
    report: str = state.get("report", "")

    results.append(check_sub_question_count(
        sub_questions,
        tuple(golden["expected_sub_questions"]),  # type: ignore[arg-type]
    ))
    results.append(check_sub_questions_non_empty(sub_questions))
    results.append(check_source_count(evaluated_sources, golden["min_sources"]))
    results.append(check_no_duplicate_urls(evaluated_sources))
    results.append(check_report_has_references(report))
    results.append(check_report_has_headings(report))
    results.append(check_report_word_count(
        report,
        golden.get("min_report_words", 200),
        golden.get("max_report_words", 3000),
    ))
    results.append(check_report_has_citations(report))
    results.extend(check_required_topics(report, golden.get("required_topics", [])))
    results.extend(check_forbidden_topics(report, golden.get("forbidden_topics", [])))

    return results
