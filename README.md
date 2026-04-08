# 🗺️ Cartograph

> Maps out knowledge so you don't have to — autonomous research agent that decomposes topics,
> searches the web, evaluates sources, and generates cited reports. Fully local, zero API keys.

---

## How It Works

The agent runs as a **LangGraph state machine** with a self-correcting retry loop. See [`docs/architecture.md`](docs/architecture.md) for a full breakdown of node responsibilities and design decisions.

1. **Planner** — Breaks your topic into 3–5 focused, non-overlapping sub-questions using Llama 3.1
2. **Searcher** — Runs sequential DuckDuckGo searches for each sub-question; deduplicates results via FAISS cosine similarity
3. **Evaluator** — Scores every source 1–5 for relevance; checks that each sub-question has enough high-quality sources. Retries search if coverage is thin (up to `max_retries` cycles)
4. **Synthesizer** — Generates a structured Markdown report with inline `[N]` citations and an auto-appended References section

---

## Quick Start

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed

```bash
ollama pull llama3.1

git clone https://github.com/surabhi/cartograph.git
cd cartograph

uv venv && source .venv/bin/activate
uv sync

python app.py
# Open http://localhost:7860
```

---

## Configuration

All tuneable parameters live in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `llama3.1` | Ollama model to use |
| `llm_timeout` | `120` | Seconds before an LLM call times out |
| `max_sub_questions` | `5` | Sub-questions the planner generates |
| `results_per_query` | `5` | DuckDuckGo results per sub-question |
| `min_relevance_score` | `3` | Minimum score (1–5) to keep a source |
| `min_sources_per_question` | `2` | Sources required per sub-question before proceeding |
| `max_retries` | `2` | Search/evaluate retry cycles if coverage is insufficient |
| `dedup_threshold` | `0.92` | Cosine similarity threshold for FAISS deduplication |
| `snippet_max_chars` | `500` | Characters of snippet passed to the synthesizer |

The Gradio UI exposes **Research Depth** (Quick / Standard / Deep), which maps to 3, 5, or 7 sub-questions.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) | Production-grade; native support for cycles and conditional routing |
| LLM | Llama 3.1 8B via [Ollama](https://ollama.com) | Fully local — zero API cost, complete privacy |
| Web search | [ddgs](https://pypi.org/project/ddgs/) | No API key required |
| Deduplication | FAISS + `all-MiniLM-L6-v2` | Semantic similarity prevents redundant sources across sub-queries |
| UI | [Gradio](https://gradio.app) | Streaming progress updates, zero frontend code |

---

## Project Structure

```
cartograph/
├── src/
│   ├── config.py               # AgentConfig — all tuneable params
│   ├── agent/
│   │   ├── state.py            # ResearchState TypedDict
│   │   ├── graph.py            # LangGraph StateGraph with retry loop
│   │   ├── nodes/
│   │   │   ├── planner.py
│   │   │   ├── searcher.py
│   │   │   ├── evaluator.py
│   │   │   └── synthesizer.py
│   │   └── prompts/            # Decoupled .txt prompt files
│   ├── search/ddg.py           # DuckDuckGo wrapper with retry backoff
│   └── retrieval/cache.py      # FAISS dedup cache (graceful fallback)
├── app.py                      # Gradio UI with streaming progress
└── tests/                      # pytest — nodes, cache, graph (happy + unhappy paths)
```

---

## Limitations & Future Work

- DuckDuckGo rate-limits aggressive parallel searches — add jitter or a search provider fallback
- Llama 3.1 8B occasionally produces verbose synthesis; a larger model improves quality
- No persistence — each session starts fresh (by design; avoids cross-session pollution)
- Future: PDF ingestion, async search, MCP tool integration, streaming token output

---

## License

MIT
