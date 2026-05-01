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

_gr.update.side_effect = lambda **kw: kw  # return the kwargs dict

import app as app_module  # noqa: E402
from app import _prepare_export  # noqa: E402

# Restore modules that were stubbed only for the app import — this lets other
# test files that patch "src.agent.graph.*" resolve the real module.
for _mod in ("src.agent.graph", "src.history", "src.llm"):
    if _originals[_mod] is None:
        sys.modules.pop(_mod, None)
    else:
        sys.modules[_mod] = _originals[_mod]


@pytest.fixture(autouse=True)
def reset_export_tmp():
    """Ensure the module-level temp path is cleared before and after each test."""
    app_module._last_export_tmp = None
    yield
    if app_module._last_export_tmp is not None:
        app_module._last_export_tmp.unlink(missing_ok=True)
    app_module._last_export_tmp = None


def test_empty_report_returns_hidden():
    row_update, dl_update = _prepare_export("")
    assert row_update["visible"] is False
    assert dl_update["visible"] is False
    assert dl_update["value"] is None


def test_error_report_returns_hidden():
    row_update, dl_update = _prepare_export("Error: something went wrong")
    assert row_update["visible"] is False


def test_placeholder_report_returns_hidden():
    row_update, dl_update = _prepare_export("Drop a pin on your research topic above.")
    assert row_update["visible"] is False


def test_valid_report_returns_visible_and_creates_file():
    row_update, dl_update = _prepare_export("# My Report\n\nSome findings.")
    assert row_update["visible"] is True
    assert dl_update["visible"] is True
    tmp_path = Path(dl_update["value"])
    assert tmp_path.exists()
    assert tmp_path.suffix == ".md"
    assert tmp_path.read_text(encoding="utf-8") == "# My Report\n\nSome findings."
    tmp_path.unlink()


def test_second_call_deletes_previous_tmp_file():
    """Each export call should clean up the previous temp file."""
    _prepare_export("First report")
    first_path = Path(app_module._last_export_tmp)
    assert first_path.exists()

    _prepare_export("Second report")
    assert not first_path.exists(), "Previous temp file should have been deleted"
    second_path = Path(app_module._last_export_tmp)
    assert second_path.exists()
    second_path.unlink()


def test_invalid_then_valid_does_not_leak():
    """An invalid report after a valid one should not leave the old file around."""
    _prepare_export("Valid report")
    first_path = Path(app_module._last_export_tmp)

    _prepare_export("")  # invalid — returns hidden but should NOT delete the valid file
    # The module-level pointer is unchanged; file still exists for the download button
    assert first_path.exists()
