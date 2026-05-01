# Cartograph — Distribution Readiness Spec

This document is a handoff spec for Claude Code. The goal is to make Cartograph
ready for public distribution on GitHub, targeting technical users who run local
LLMs. All tasks are self-contained and ordered by priority.

---

## Context

Cartograph is a local-first AI research agent built with LangGraph + Gradio. It
runs fully on-device via Ollama with no API keys required by default. The
distribution model is open-source GitHub release — users clone, set up, and run
it themselves.

The codebase is clean (44/44 tests passing, ruff 0 errors). The gaps are almost
entirely in setup experience and documentation, not in the code itself.

---

## Task 1 — Setup health check script

**File to create:** `scripts/check.py`

**Purpose:** A single script users can run before `python app.py` to verify all
dependencies are in place. Should print a clear pass/fail for each check and
give an actionable fix for every failure.

**Checks to implement, in order:**

1. **Python version** — verify `sys.version_info >= (3, 11)`. Fail message:
   `Python 3.11+ required. You have X.Y. Install from https://python.org`

2. **Ollama reachability** — make an HTTP GET to `http://localhost:11434/api/tags`
   with a 3-second timeout using `urllib.request` (stdlib only, no extra deps).
   Fail message:
   ```
   Ollama is not running.
   Fix: open a new terminal and run: ollama serve
   ```

3. **Required model present** — from the same `/api/tags` response, check whether
   at least one model is listed. If none:
   ```
   No models found in Ollama.
   Fix: ollama pull llama3.1
   ```
   If models exist, print which ones were found.

4. **Core dependencies importable** — try importing `langgraph`, `gradio`,
   `langchain_ollama`, `ddgs`, `sentence_transformers`. For each import failure:
   ```
   Missing dependency: <package>
   Fix: uv sync
   ```

5. **Optional dependencies** — try importing `faiss`, `langchain_anthropic`,
   `langchain_openai`, `tavily`. For each missing one, print a non-fatal warning:
   ```
   Optional: faiss not installed — URL-based dedup will be used instead of semantic
   Optional: langchain_anthropic not installed — Anthropic provider unavailable
   ```

**Output format:** Each check should print a line like:
```
✓  Python 3.11.9
✓  Ollama running (3 models found: llama3.1, mistral, phi3)
✓  Core dependencies installed
⚠  faiss not installed (optional — semantic dedup disabled)
✗  Ollama not running — run: ollama serve
```

Exit with code 0 if all required checks pass (warnings are OK), exit code 1 if
any required check fails.

**Important:** The script must use only stdlib for the Ollama check so it works
before `uv sync` has been run.

---

## Task 2 — Update README.md Quick Start section

**File to edit:** `README.md`

The existing Quick Start section assumes `uv` is installed and Ollama is already
running. Update it to remove those assumptions.

**Replace the existing Quick Start section with:**

```markdown
## Quick Start

### 1. Install prerequisites

**Ollama** (runs your local LLM):
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download the installer from https://ollama.com/download
```

Then pull the default model (one-time, ~5 GB):
```bash
ollama pull llama3.1
```

**uv** (Python package manager):
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and install

```bash
git clone https://github.com/surabhi/cartograph.git
cd cartograph
uv venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

### 3. Verify setup (optional but recommended)

```bash
python scripts/check.py
```

### 4. Run

```bash
python app.py
# Open http://localhost:7860
```
```

**Also add after the Quick Start section:**

```markdown
## Using cloud providers (optional)

Cartograph works out of the box with Ollama. To use Anthropic or OpenAI instead,
install the relevant extra and enter your API key in the UI:

```bash
uv add langchain-anthropic   # for Anthropic Claude
uv add langchain-openai      # for OpenAI GPT models
uv add tavily-python         # for Tavily search (optional alternative to DuckDuckGo)
```

Cloud providers send queries to external APIs — your research topics will be
visible to those providers. Use Ollama if privacy matters.
```

---

## Task 3 — Add TROUBLESHOOTING.md

**File to create:** `docs/TROUBLESHOOTING.md`

Create a troubleshooting guide covering the 7 most common failure scenarios.
Each entry should have: symptom, cause, and exact fix.

**Entries to include:**

### 1. `Connection refused` or `Failed to connect to Ollama`
- **Symptom:** Error when starting app or during a research run
- **Cause:** Ollama process is not running
- **Fix:**
  ```bash
  ollama serve        # start Ollama in a separate terminal
  # then rerun: python app.py
  ```

### 2. `unknown model` or `model not found`
- **Symptom:** Error during planning or synthesis step
- **Cause:** The selected model hasn't been pulled to local Ollama
- **Fix:**
  ```bash
  ollama pull llama3.1          # or whichever model you selected
  ollama list                   # verify it appears
  ```

### 3. `Port 7860 is already in use`
- **Symptom:** App fails to start
- **Cause:** Another Gradio app or service is using port 7860
- **Fix:** Set a different port in `.env`:
  ```
  GRADIO_PORT=7861
  ```
  Then restart `python app.py`.

### 4. `ModuleNotFoundError` for any package
- **Symptom:** Import error on startup
- **Cause:** Virtual environment not activated or `uv sync` not run
- **Fix:**
  ```bash
  source .venv/bin/activate     # Windows: .venv\Scripts\activate
  uv sync
  python app.py
  ```

### 5. LLM response is very slow or times out
- **Symptom:** Research hangs at "Charting the route" or times out after 120s
- **Cause:** Llama 3.1 8B is too large for available RAM/VRAM
- **Fix options:**
  - Use a smaller model: `ollama pull phi3` or `ollama pull mistral`
  - Select the smaller model from the Model dropdown in the UI
  - Increase the timeout in `src/config.py`: `llm_timeout = 300`

### 6. DuckDuckGo returns no results or rate-limit error
- **Symptom:** "0 sources surveyed" or DDGSException in logs
- **Cause:** DuckDuckGo rate-limited the request (common with quick repeated runs)
- **Fix:** Wait 30–60 seconds before retrying. For sustained use, switch to Tavily:
  ```bash
  uv add tavily-python
  # Get a free API key at https://tavily.com
  # Enter it in the Tavily API Key field in the UI
  ```

### 7. `langchain_anthropic` or `langchain_openai` not found
- **Symptom:** Error when switching Provider to Anthropic or OpenAI in the UI
- **Cause:** Cloud provider extras are not installed by default
- **Fix:**
  ```bash
  uv add langchain-anthropic   # for Anthropic
  uv add langchain-openai      # for OpenAI
  ```

**Add a link to this file from README.md** — append to the end of the
Quick Start section:
```markdown
Having trouble? See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
```

---

## Task 4 — Add input validation to app.py

**File to edit:** `app.py`

Currently an empty query shows "Drop a pin on your research topic above." which
is fine, but there's no minimum length check — a single character query will
run the full pipeline and produce a poor result.

**In the `research()` function, replace the existing empty check:**

```python
# existing
if not query:
    yield "Drop a pin on your research topic above.", ""
    return
```

**With:**

```python
if not query:
    yield "Drop a pin on your research topic above.", ""
    return
if len(query) < 10:
    yield "Your query is too short — try describing your topic in more detail.", ""
    return
if len(query) > 500:
    yield "Query too long (500 character max) — try a more focused topic.", ""
    return
```

---

## Task 5 — Render the architecture diagram

**Context:** `docs/architecture.mermaid` exists as source but has never been
compiled to a PNG. The architecture.md references a diagram that isn't actually
embedded.

**Steps:**

1. Check if `mmdc` (mermaid CLI) is available: `which mmdc`
2. If not, install it: `npm install -g @mermaid-js/mermaid-cli`
3. Render: `mmdc -i docs/architecture.mermaid -o docs/architecture.png`
4. Edit `docs/architecture.md` to embed the image — find the line that references
   the mermaid file and add below it:
   ```markdown
   ![Cartograph architecture](architecture.png)
   ```

---

## Task 6 — Add a Makefile for common commands

**File to create:** `Makefile`

New contributors and users will expect standard `make` targets. This reduces the
"what command do I run?" friction significantly.

```makefile
.PHONY: install run check test lint clean

install:
	uv venv && . .venv/bin/activate && uv sync

run:
	. .venv/bin/activate && python app.py

check:
	. .venv/bin/activate && python scripts/check.py

test:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/ -v

lint:
	. .venv/bin/activate && ruff check src/ app.py

clean:
	rm -rf .venv __pycache__ src/__pycache__ history.db
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
```

Update the README.md development section to show `make install`, `make run`,
`make test` as the canonical commands.

---

## Task 7 — Add .env.example entries for new config

**File to edit:** `.env.example`

The existing `.env.example` covers `OLLAMA_HOST`, `OLLAMA_MODEL`, and
`GRADIO_PORT`. Verify these entries are present and correct, and add any missing
ones:

```bash
# Cartograph environment configuration
# Copy to .env and edit — all values are optional

# Ollama
OLLAMA_HOST=http://localhost:11434   # change if Ollama runs on another machine
OLLAMA_MODEL=llama3.1               # default model loaded in the UI

# Gradio
GRADIO_PORT=7860                    # change if port 7860 is taken

# Cloud providers (only needed if using Anthropic or OpenAI in the UI)
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...

# Search (only needed if using Tavily in the UI)
# TAVILY_API_KEY=tvly-...
```

---

## Task 8 — Update README.md Limitations section

**File to edit:** `README.md`

The existing Limitations section mentions "No persistence — each session starts
fresh (by design)." This is incorrect — the app does persist history to
`history.db`. Update this entry:

**Find and replace:**
```
- No persistence — each session starts fresh (by design; avoids cross-session pollution)
```
**With:**
```
- Research history is stored locally in `history.db` — not synced across machines
```

Also update the Future Work list to reflect the current state more accurately.
Replace the existing future work bullets with:

```markdown
**Future ideas:**
- PDF / document ingestion as a research source
- Streaming token output (currently streams by node, not by token)
- MCP tool integration for custom search providers
- Export to PDF or DOCX in addition to Markdown
- Re-run eval suite after any synthesizer prompt changes to track factual grounding score
```

---

## Task 9 — Add a CONTRIBUTING.md

**File to create:** `CONTRIBUTING.md`

Keep this short — one page is enough for v0.1.

```markdown
# Contributing to Cartograph

Thanks for your interest. Here's how to get set up and what's useful to know.

## Setup

```bash
git clone https://github.com/surabhi/cartograph.git
cd cartograph
uv venv && source .venv/bin/activate
uv sync
```

## Running tests

```bash
PYTHONPATH=. pytest tests/ -v       # all 44 tests
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
```

---

## Task 10 — Verify tests still pass after all changes

After completing all tasks above, run the full test suite and confirm it is still
44/44:

```bash
PYTHONPATH=. pytest tests/ -v
ruff check src/ app.py
python scripts/check.py
```

If any tests fail, fix them before considering the spec complete. Do not modify
test assertions to make tests pass — fix the underlying code.

---

## Completion checklist

- [ ] `scripts/check.py` created and executable
- [ ] README.md Quick Start updated with uv + Ollama install steps
- [ ] README.md cloud providers section added
- [ ] README.md links to TROUBLESHOOTING.md
- [ ] README.md Limitations section corrected
- [ ] `docs/TROUBLESHOOTING.md` created with 7 entries
- [ ] `app.py` query length validation added
- [ ] `docs/architecture.png` rendered and embedded in architecture.md
- [ ] `Makefile` created with install / run / check / test / lint / clean targets
- [ ] `.env.example` verified and updated
- [ ] `CONTRIBUTING.md` created
- [ ] 44/44 tests passing
- [ ] `ruff check` 0 errors
