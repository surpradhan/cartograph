"""Tests for the export half of app._after_research.

The export logic used to live in a dedicated _prepare_export() function; PR #10
folded it into _after_research(), which now returns updates for
[history_dropdown, export_row, dl_btn] and reads the completed report from the
module-level _last_completed_report cache.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Stub out heavy / unavailable dependencies before importing app.
# Only set stubs that aren't already in sys.modules so we don't corrupt the
# real modules that other test files need at runtime. We restore non-persistent
# stubs after app is imported so patch("src.agent.graph...") works elsewhere.
_STUB_MODS = (
    "ddgs",
    "ddgs.exceptions",
    "gradio",
    "src.agent.graph",
    "src.history",
    "src.llm",
)
_originals: dict = {}
for _mod in _STUB_MODS:
    _originals[_mod] = sys.modules.get(_mod)
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# gradio needs a real-ish update() for the return-value assertions
import gradio as _gr  # noqa: E402  (already mocked above)

if isinstance(_gr, MagicMock):
    _gr.update.side_effect = lambda **kw: kw  # return the kwargs dict

import app as app_module  # noqa: E402
from app import _after_research  # noqa: E402

# Restore modules that were stubbed only for the app import — this lets other
# test files that patch "src.agent.graph.*" resolve the real module.
for _mod in ("src.agent.graph", "src.history", "src.llm"):
    if _originals[_mod] is None:
        sys.modules.pop(_mod, None)
    else:
        sys.modules[_mod] = _originals[_mod]


@pytest.fixture(autouse=True)
def isolate_app_state(monkeypatch):
    """Reset export/report module state and keep history I/O out of the tests.

    app imported save_run/load_recent by name, so patch the bound names on the
    app module itself — this works whether or not src.history was stubbed above.
    """
    monkeypatch.setattr(app_module, "save_run", MagicMock())
    monkeypatch.setattr(app_module, "load_recent", lambda n: [])
    app_module._last_export_tmp = None
    app_module._last_completed_report = ""
    yield
    if app_module._last_export_tmp is not None:
        app_module._last_export_tmp.unlink(missing_ok=True)
    app_module._last_export_tmp = None
    app_module._last_completed_report = ""


def _run_after_research(report: str):
    app_module._last_completed_report = report
    return _after_research("test query", "Standard", "test-model")


def test_empty_report_returns_hidden():
    _hist, row_update, dl_update = _run_after_research("")
    assert row_update["visible"] is False
    assert dl_update["visible"] is False
    assert dl_update["value"] is None
    app_module.save_run.assert_not_called()


def test_error_report_returns_hidden():
    _hist, row_update, _dl = _run_after_research("Error: something went wrong")
    assert row_update["visible"] is False
    app_module.save_run.assert_not_called()


def test_drop_a_pin_status_returns_hidden():
    # The "Drop a pin" string is the empty-query status message; it must be
    # rejected by the startswith("Drop a pin") guard.
    _hist, row_update, _dl = _run_after_research(
        "Drop a pin on your research topic above."
    )
    assert row_update["visible"] is False
    app_module.save_run.assert_not_called()


def test_report_placeholder_returns_hidden():
    # The report-box placeholder ("*Plant a pin...*") is caught by the separate
    # `report != _REPORT_PLACEHOLDER` guard, not the startswith check.
    _hist, row_update, _dl = _run_after_research(app_module._REPORT_PLACEHOLDER)
    assert row_update["visible"] is False
    app_module.save_run.assert_not_called()


def test_valid_report_returns_visible_and_creates_file():
    _hist, row_update, dl_update = _run_after_research("# My Report\n\nSome findings.")
    assert row_update["visible"] is True
    assert dl_update["visible"] is True
    tmp_path = Path(dl_update["value"])
    assert tmp_path.exists()
    assert tmp_path.suffix == ".md"
    assert tmp_path.read_text(encoding="utf-8") == "# My Report\n\nSome findings."
    app_module.save_run.assert_called_once_with(
        "test query", "Standard", "test-model", "# My Report\n\nSome findings."
    )
    tmp_path.unlink()


def test_second_call_deletes_previous_tmp_file():
    """Each export call should clean up the previous temp file."""
    _run_after_research("First report")
    first_path = Path(app_module._last_export_tmp)
    assert first_path.exists()

    _run_after_research("Second report")
    assert not first_path.exists(), "Previous temp file should have been deleted"
    second_path = Path(app_module._last_export_tmp)
    assert second_path.exists()
    second_path.unlink()


def test_invalid_then_valid_does_not_leak():
    """An invalid report after a valid one should not delete the valid file."""
    _run_after_research("Valid report")
    first_path = Path(app_module._last_export_tmp)

    _run_after_research("")  # invalid — returns hidden but must NOT delete the file
    # The module-level pointer is unchanged; file still exists for the download button
    assert first_path.exists()
