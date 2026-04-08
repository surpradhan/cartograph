# Cartograph — Architecture

## Overview

Cartograph is a multi-node research agent built on LangGraph. A single user query
flows through four sequential nodes, with a conditional retry loop between the
Evaluator and Searcher.

## Node Responsibilities

**Planner** (`src/agent/nodes/planner.py`)
Receives the raw user query and calls Llama 3.1 to decompose it into 3–5 focused
sub-questions. Output is stored in `state.sub_questions`.

**Searcher** (`src/agent/nodes/searcher.py`)
Runs sequential DuckDuckGo searches for each sub-question (ddgs uses primp, a Rust
HTTP library that is not thread-safe). A fresh FAISS source cache is created on every
call so that retry cycles are not blocked by results from previous cycles. Before
adding each result it checks the cache for near-duplicate snippets (cosine similarity
> 0.92). Output: `state.search_results`.

**Evaluator** (`src/agent/nodes/evaluator.py`)
Scores each source on a 1–5 relevance scale using the configured LLM (scoring runs
in parallel via `ThreadPoolExecutor`). Sources below the threshold (default: 3) are
discarded. If at least `min_sources_per_question` high-quality sources exist per
sub-question, `coverage_sufficient` is set to True and the graph proceeds to
synthesis. Otherwise, if `retry_count <= max_retries`, the graph loops back to the
Searcher. Once `retry_count > max_retries` the graph proceeds to synthesis regardless.

**Synthesizer** (`src/agent/nodes/synthesizer.py`)
Generates a structured Markdown research report from the evaluated sources.
Every factual claim is backed by an inline citation `[N]` linking to a source URL.
A references section is appended automatically.

## Data Flow

```
query → Planner → sub_questions
sub_questions → Searcher → search_results (deduped via FAISS)
search_results → Evaluator → evaluated_sources, coverage_sufficient, retry_count
evaluated_sources → Synthesizer → report
```

See `docs/architecture.mermaid` for the visual diagram.

## Design Decisions

| Decision | Rationale |
|---|---|
| LangGraph StateGraph | Supports cycles (retry loop) and conditional routing natively |
| TypedDict state | Explicit schema; catches missing keys at development time |
| FAISS + MiniLM | Lightweight dedup without a full vector database |
| Prompt files (.txt) | Prompts are decoupled from code; easy to iterate without touching Python |
| Dataclass config | Single source of truth for all tuneable parameters |
