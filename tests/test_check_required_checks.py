"""Tests for the drift-guard's branch-protection error handling.

These cover the riskiest logic in check_required_checks.py: the live API call
must fail *closed* on a genuine 404 (protection absent) but skip gracefully —
returning the _UNREACHABLE sentinel — on any transient/auth/network/parse
failure, so an unrelated GitHub hiccup never blocks merges.
"""

import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

# The script lives under .github/scripts (not a package), so load it by path.
_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github" / "scripts" / "check_required_checks.py"
)
_spec = importlib.util.spec_from_file_location("check_required_checks", _SCRIPT)
crc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crc)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.github.com", code=code, msg="x", hdrs=None, fp=None
    )


def test_404_fails_closed_with_empty_set(monkeypatch):
    def boom(*_a, **_k):
        raise _http_error(404)

    monkeypatch.setattr(crc.urllib.request, "urlopen", boom)
    result = crc.contexts_from_branch_protection("tok", "owner/repo")
    assert result == set()  # fail closed → mismatch against a non-empty ci.yml


@pytest.mark.parametrize("code", [401, 403, 429, 500, 503])
def test_non_404_http_errors_skip(monkeypatch, code):
    def boom(*_a, **_k):
        raise _http_error(code)

    monkeypatch.setattr(crc.urllib.request, "urlopen", boom)
    assert crc.contexts_from_branch_protection("tok", "owner/repo") is crc._UNREACHABLE


def test_urlerror_skips(monkeypatch):
    def boom(*_a, **_k):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr(crc.urllib.request, "urlopen", boom)
    assert crc.contexts_from_branch_protection("tok", "owner/repo") is crc._UNREACHABLE


@pytest.mark.parametrize("exc", [TimeoutError("read timed out"), ConnectionResetError()])
def test_bare_os_errors_skip(monkeypatch, exc):
    def boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(crc.urllib.request, "urlopen", boom)
    assert crc.contexts_from_branch_protection("tok", "owner/repo") is crc._UNREACHABLE


def test_malformed_200_body_skips(monkeypatch):
    """A 200 response whose body isn't JSON must skip, not crash."""

    class _Resp:
        def read(self, *_a):
            return b"<html>incident</html>"

        def __enter__(self):
            return io.BytesIO(b"<html>incident</html>")

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(crc.urllib.request, "urlopen", lambda *_a, **_k: _Resp())
    assert crc.contexts_from_branch_protection("tok", "owner/repo") is crc._UNREACHABLE


def test_valid_response_returns_contexts(monkeypatch):
    payload = {"required_status_checks": {"contexts": ["Lint", "Tests (3.11)"]}}

    class _Resp:
        def __enter__(self):
            return io.BytesIO(json.dumps(payload).encode())

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(crc.urllib.request, "urlopen", lambda *_a, **_k: _Resp())
    assert crc.contexts_from_branch_protection("tok", "owner/repo") == {
        "Lint",
        "Tests (3.11)",
    }


def test_workflow_and_contributing_agree():
    """The repo's own ci.yml and CONTRIBUTING.md must be in sync."""
    assert crc.contexts_from_workflow() == crc.contexts_from_contributing()
