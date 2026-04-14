"""Tests for deterministic eval judge."""
import pytest
from evals.judges.deterministic import check_required_topics


def test_string_topic_found():
    results = check_required_topics("The Bronze Age collapse involved many civilizations.", ["civilization"])
    assert results[0].passed is True


def test_string_topic_missing():
    results = check_required_topics("The Bronze Age collapse reshaped societies.", ["civilization"])
    assert results[0].passed is False


def test_string_topic_case_insensitive():
    results = check_required_topics("CIVILIZATION declined rapidly.", ["civilization"])
    assert results[0].passed is True


def test_alias_list_matches_first_alias():
    results = check_required_topics(
        "Several civilizations collapsed around 1200 BCE.",
        [["civilization", "societies", "cultures"]],
    )
    assert results[0].passed is True


def test_alias_list_matches_second_alias():
    results = check_required_topics(
        "Bronze Age societies vanished almost overnight.",
        [["civilization", "societies", "cultures"]],
    )
    assert results[0].passed is True


def test_alias_list_fails_when_none_match():
    results = check_required_topics(
        "Trade routes were disrupted.",
        [["civilization", "societies", "cultures"]],
    )
    assert results[0].passed is False


def test_alias_list_label_uses_slash_joined_names():
    results = check_required_topics(
        "cultures thrived",
        [["civilization", "societies", "cultures"]],
    )
    assert results[0].name == "topic_coverage:civilization/societies/cultures"


def test_mixed_string_and_alias_list():
    report = "Bronze Age societies collapsed. The Mediterranean region was affected."
    results = check_required_topics(
        report,
        ["Bronze Age", ["civilization", "societies"], "Mediterranean"],
    )
    assert all(r.passed for r in results)
