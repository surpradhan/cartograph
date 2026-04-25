import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import src.history as history_module
from src.history import load_by_id, load_recent, save_run


@pytest.fixture(autouse=True)
def tmp_db(tmp_path):
    """Redirect history.db to a temp path for every test."""
    db = tmp_path / "test_history.db"
    with patch.object(history_module, "_DB_PATH", db):
        yield db


def test_save_and_load_recent():
    save_run("What is quantum computing?", "Standard (5)", "llama3.1", "Report body here.")
    runs = load_recent()
    assert len(runs) == 1
    assert runs[0]["query"] == "What is quantum computing?"
    assert runs[0]["model"] == "llama3.1"
    assert runs[0]["report"] == "Report body here."


def test_load_recent_returns_newest_first():
    save_run("Query A", "Quick (3)", "llama3.1", "Report A")
    save_run("Query B", "Deep (7)", "llama3.1", "Report B")
    runs = load_recent()
    assert runs[0]["query"] == "Query B"
    assert runs[1]["query"] == "Query A"


def test_load_recent_respects_limit():
    for i in range(15):
        save_run(f"Query {i}", "Standard (5)", "llama3.1", f"Report {i}")
    runs = load_recent(limit=5)
    assert len(runs) == 5


def test_load_by_id_returns_correct_report():
    save_run("First query", "Standard (5)", "llama3.1", "First report")
    save_run("Second query", "Quick (3)", "llama3.1", "Second report")
    runs = load_recent()
    second_id = runs[0]["id"]  # newest first
    assert load_by_id(second_id) == "Second report"


def test_load_by_id_returns_empty_for_missing():
    assert load_by_id(9999) == ""


def test_load_recent_returns_empty_on_no_runs():
    assert load_recent() == []


def test_save_run_is_silent_on_error():
    with patch.object(history_module, "_DB_PATH", Path("/nonexistent/dir/history.db")):
        save_run("query", "depth", "model", "report")  # should not raise
