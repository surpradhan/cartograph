## Description

<!-- What does this PR do? One or two sentences. -->

## Why

<!-- Motivation / problem being solved. -->

## Changes

<!-- Bullet list of the concrete changes. -->

-

## How to Test

<!-- Commands or steps a reviewer can run to verify. -->

```bash
PYTHONPATH=. pytest tests/ -v
ruff check src/ app.py
```

## Screenshots / Demo

<!-- For UI changes, before/after screenshots or a short clip. Delete if N/A. -->

## Related Issues

<!-- e.g. Closes #123 -->

## Checklist

- [ ] Tests pass locally (`PYTHONPATH=. pytest tests/ -v`)
- [ ] Lint passes (`ruff check src/ app.py`)
- [ ] Tests added/updated for new behavior
- [ ] Docs updated (README/CONTRIBUTING/docs) where relevant
- [ ] No breaking changes, or they are documented above
- [ ] Commit messages follow `<type>: <subject>` format
- [ ] Branch is up to date with `main`
- [ ] If CI jobs changed: `ci.yml`, the CONTRIBUTING required-checks block, and
      branch protection updated together

## Type of Change

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `refactor` — no behavior change
- [ ] `perf` — performance improvement
- [ ] `chore` — tooling / maintenance
