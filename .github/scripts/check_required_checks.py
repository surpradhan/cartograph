#!/usr/bin/env python3
"""Drift-guard: keep the three lists of required status checks in lock-step.

Sources compared:
  1. `.github/workflows/ci.yml` — each job's `name:`, with matrix jobs
     expanded to GitHub's "Name (value)" / "Name (v1, v2)" contexts.
  2. `CONTRIBUTING.md` — the backtick-quoted names between the
     `<!-- required-checks:start -->` / `<!-- required-checks:end -->` markers.
  3. (optional) Live branch protection on `main`, fetched from the GitHub API
     when the `CHECKS_SYNC_TOKEN` env var holds an admin token. Skipped
     otherwise, so (1)-vs-(2) always runs credential-free (works on forks).

Exits non-zero on any mismatch and prints a diff. Run locally with:
    python .github/scripts/check_required_checks.py
"""

import itertools
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"
PROTECTED_BRANCH = "main"
MARKER_START = "<!-- required-checks:start -->"
MARKER_END = "<!-- required-checks:end -->"


def contexts_from_workflow() -> set[str]:
    """Derive the status-check contexts GitHub will report for ci.yml."""
    data = yaml.safe_load(WORKFLOW.read_text())
    contexts: set[str] = set()
    for job_id, job in data.get("jobs", {}).items():
        name = job.get("name", job_id)
        matrix = (job.get("strategy") or {}).get("matrix")
        if not matrix:
            contexts.add(name)
            continue
        axes = {k: v for k, v in matrix.items() if k not in ("include", "exclude")}
        if "include" in matrix or "exclude" in matrix:
            sys.exit(
                f"error: job '{job_id}' uses matrix include/exclude, which this "
                "script does not expand — extend contexts_from_workflow() first."
            )
        if "${{" in name:
            sys.exit(
                f"error: job '{job_id}' interpolates matrix values in its name; "
                "use a plain name and let GitHub append the matrix suffix."
            )
        for combo in itertools.product(*axes.values()):
            suffix = ", ".join(str(v) for v in combo)
            contexts.add(f"{name} ({suffix})")
    return contexts


def contexts_from_contributing() -> set[str]:
    """Parse the backtick-quoted check names from the CONTRIBUTING block."""
    text = CONTRIBUTING.read_text()
    try:
        start = text.index(MARKER_START) + len(MARKER_START)
        end = text.index(MARKER_END)
    except ValueError:
        sys.exit(f"error: required-checks markers not found in {CONTRIBUTING.name}")
    return set(re.findall(r"`([^`]+)`", text[start:end]))


_UNREACHABLE = object()  # sentinel: the live check could not be performed


def contexts_from_branch_protection(token: str, repo: str):
    """Fetch live required contexts.

    Returns a set of context names on success, or the _UNREACHABLE sentinel
    when the API could not be queried (network/rate-limit/5xx/auth) — callers
    treat that as "skip the live check" rather than a mismatch, so an unrelated
    GitHub hiccup never blocks merges. A genuine 404 (no protection on the
    branch) returns an empty set, which fails closed against a non-empty ci.yml.
    """
    url = f"https://api.github.com/repos/{repo}/branches/{PROTECTED_BRANCH}/protection"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # GitHub returns 404 both when protection is genuinely absent AND
            # when the token lacks admin-read scope — they're indistinguishable
            # here. Treat as an empty set (fails closed): if you set the token
            # you intend protection to exist, so a real mismatch surfaces.
            print(f"warning: branch protection on '{PROTECTED_BRANCH}' returned "
                  f"HTTP 404 — either it is not configured, or CHECKS_SYNC_TOKEN "
                  f"lacks Administration:read scope. Treating as empty set.")
            return set()
        # 401/403/429/5xx etc. — transient or auth issues unrelated to drift.
        # Don't fail the build on these; the credential-free ci-vs-docs check
        # is the real guard and has already run.
        print(f"warning: branch-protection API call failed (HTTP {exc.code}) — "
              f"skipping live branch-protection check")
        return _UNREACHABLE
    except urllib.error.URLError as exc:
        print(f"warning: could not reach the GitHub API ({exc.reason}) — "
              f"skipping live branch-protection check")
        return _UNREACHABLE
    except (TimeoutError, OSError, json.JSONDecodeError) as exc:
        # Catch-all for the remaining best-effort failure modes that aren't
        # URLError subclasses: a bare read-timeout (TimeoutError), low-level
        # socket errors (ConnectionResetError, ...), and a 200 with a non-JSON
        # body (proxy/incident HTML, truncated response). None of these signal
        # drift, so skip rather than fail the build. URLError is itself an
        # OSError subclass, so it must be (and is) handled above this.
        print(f"warning: branch-protection check failed ({exc}) — "
              f"skipping live branch-protection check")
        return _UNREACHABLE
    checks = data.get("required_status_checks") or {}
    return set(checks.get("contexts") or [])


def report_diff(label_a: str, a: set[str], label_b: str, b: set[str]) -> bool:
    ok = True
    for missing in sorted(a - b):
        print(f"MISMATCH: '{missing}' is in {label_a} but missing from {label_b}")
        ok = False
    for stale in sorted(b - a):
        print(f"MISMATCH: '{stale}' is in {label_b} but not in {label_a}")
        ok = False
    return ok


def main() -> int:
    ci = contexts_from_workflow()
    docs = contexts_from_contributing()
    print(f"ci.yml contexts:        {sorted(ci)}")
    print(f"CONTRIBUTING contexts:  {sorted(docs)}")

    ok = report_diff("ci.yml", ci, "CONTRIBUTING.md", docs)

    token = os.environ.get("CHECKS_SYNC_TOKEN")
    # In CI this is always set; the default only matters for local `python
    # .github/scripts/...` runs. A fork running locally without exporting
    # GITHUB_REPOSITORY would query the upstream repo — set it if that's wrong.
    repo = os.environ.get("GITHUB_REPOSITORY", "surpradhan/cartograph")
    if token:
        live = contexts_from_branch_protection(token, repo)
        if live is _UNREACHABLE:
            pass  # already warned; the live check is a bonus, not a gate
        else:
            print(f"branch-protection contexts: {sorted(live)}")
            # Anchored on ci (not docs) because ci-vs-docs is already enforced
            # above; with that holding, ci == live implies docs == live.
            ok = report_diff("ci.yml", ci, "branch protection", live) and ok
    else:
        print("CHECKS_SYNC_TOKEN not set — skipping live branch-protection check")

    if not ok:
        print(
            "\nFix: when adding/renaming/removing a CI job, update ci.yml, the "
            "CONTRIBUTING.md required-checks block, and the branch-protection "
            "required contexts in the same PR."
        )
        return 1
    print("OK: required checks are in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
