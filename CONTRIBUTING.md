# Contributing to Cartograph

Thanks for your interest. Here's how to get set up and what's useful to know.

## Setup

```bash
git clone https://github.com/surpradhan/cartograph.git
cd cartograph
uv venv && source .venv/bin/activate
uv sync
```

## Running tests

```bash
PYTHONPATH=. pytest tests/ -v       # all 102 tests
PYTHONPATH=. pytest tests/test_evaluator.py -v   # single file
```

## Linting

```bash
ruff check src/ app.py
```

All PRs must pass `pytest` and `ruff check` with zero errors.

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

Please include: OS, Python version, Ollama version (`ollama --version`), the
model you were using, and the full error message or log output.
