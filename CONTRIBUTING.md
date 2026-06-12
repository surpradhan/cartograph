# Contributing to Cartograph

Thanks for your interest. Here's how to get set up, how changes land, and
what's useful to know.

## Setup

```bash
git clone https://github.com/surpradhan/cartograph.git
cd cartograph
uv venv && source .venv/bin/activate
uv sync --extra dev
```

## Running tests

```bash
PYTHONPATH=. pytest tests/ -v                    # full suite
PYTHONPATH=. pytest tests/test_evaluator.py -v   # single file
```

## Linting

```bash
ruff check src/ app.py
```

All PRs must pass `pytest` and `ruff check` with zero errors.

## How changes land

`main` is protected and PR-only — no direct pushes, no force-pushes, no branch
deletion. These rules apply to maintainers (and AI agents) too.

Every change follows the same path:

1. Branch off `main` (e.g. `feat/your-change`).
2. Commit your work (format below).
3. Open a PR against `main` using the PR template.
4. CI must be green, the branch up to date with `main`, and **all review
   conversations resolved**.
5. A maintainer reviews; **approval is required to merge**.
6. **Squash-merge** — history on `main` stays linear, no merge commits.

### Required status checks

The canonical list of required checks. This block is machine-read by
`.github/scripts/check_required_checks.py` (the `Required checks in sync` CI
job), which fails if it drifts from `.github/workflows/ci.yml` or from the
branch-protection settings on `main`.

<!-- required-checks:start -->
- `Lint`
- `Tests (3.11)`
- `Tests (3.12)`
- `Smoke`
- `Required checks in sync`
<!-- required-checks:end -->

**Drift-guard rule:** whenever you add, rename, or remove a CI job, update
`ci.yml`, the list above, AND the branch-protection required contexts **in the
same PR**.

### Branch protection on `main`

Configured in the GitHub UI (Settings → Branches):

- Require a pull request before merging, with 1 approval
- Require status checks to pass (the list above) and branches to be up to date
- Require conversation resolution before merging
- Require linear history
- Block force pushes and branch deletion
- Rules enforced for administrators

## Commit message format

```
<type>: <subject>

<body (optional)>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.
Subject in the imperative mood ("add X", not "added X"). No attribution
footers.

## Code review

- Automated checks must pass **and** a maintainer must review every PR — a
  green CI run is necessary but not sufficient.
- Approval is required to merge (enforced by branch protection).
- Address review comments, then re-request review.
- We aim to review within 48 hours.

## Releases

Cartograph doesn't currently publish packages or images — releases are git
tags plus GitHub Releases. If we start publishing artifacts (PyPI, Docker),
the release workflow must live in a separate `release-*.yml` (so the
drift-guard ignores it), verify the tagged commit is an ancestor of
`origin/main`, gate publishing behind a GitHub Environment with required
reviewers, keep publish tokens as environment secrets, and publish with
provenance where supported.

## AI-agent working conventions

These apply to any AI agent (Claude Code etc.) working in this repo:

- **Never self-merge to `main`.** The PR author — human or agent — is never
  also the reviewer/merger. Stop at "PR open, CI green, awaiting review"; a
  separate session or person reviews and merges. CI-green ≠ reviewed.
- **Contributor grace period:** don't self-implement issues opened for
  external contributors (e.g. `good-first-issue`) for 3 days unless urgent.
- **Contributor PR reviews:** leave feedback and let the contributor apply
  it; only push direct fixes to their branch when the owner explicitly says
  so. Credit them on merge and comment on the linked issue.
- **Drift-guard discipline:** CI workflow, the required-checks block above,
  and branch protection are edited together, in the same PR.

## Project structure

- `src/agent/nodes/` — the four agent nodes (planner, searcher, evaluator, synthesizer)
- `src/agent/prompts/` — LLM prompts as plain `.txt` files; iterate here without touching Python
- `src/config.py` — all tuneable parameters in one place
- `app.py` — Gradio UI; keep this thin; business logic belongs in `src/`
- `evals/` — golden set evaluation harness; run with `python evals/run_evals.py`

## Key gotchas

1. `ddgs` uses a Rust HTTP backend that is not thread-safe — do not move searches
   back to `ThreadPoolExecutor`
2. `SourceCache` must be created inside `run_searcher()`, not at module level —
   a shared cache blocks retry cycles by flagging new results as duplicates
3. Retry semantics use `>` not `>=` — see `src/agent/graph.py` and the comment
   in `CLAUDE.md` for the full explanation

## Opening issues

Please use the issue templates. For bugs, include: OS, Python version, Ollama
version (`ollama --version`), the model you were using, and the full error
message or log output.
