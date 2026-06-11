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


def contexts_from_branch_protection(token: str, repo: str) -> set[str] | None:
    """Fetch live required contexts; returns None if inaccessible."""
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
            # 404 = branch protection not configured (or token lacks scope).
            print(f"warning: no branch protection found on '{PROTECTED_BRANCH}' "
                  f"(HTTP 404) — treating as empty set")
            return set()
        sys.exit(f"error: branch-protection API call failed: HTTP {exc.code}")
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
    repo = os.environ.get("GITHUB_REPOSITORY", "surpradhan/cartograph")
    if token:
        live = contexts_from_branch_protection(token, repo)
        print(f"branch-protection contexts: {sorted(live)}")
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
