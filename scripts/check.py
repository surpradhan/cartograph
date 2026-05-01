#!/usr/bin/env python3
"""Pre-flight check — run before `python app.py` to verify all dependencies."""
import json
import os
import sys
import urllib.error
import urllib.request

PASS = "\u2713 "
WARN = "\u26a0  "
FAIL = "\u2717  "

exit_code = 0


def fail(msg: str) -> None:
    global exit_code
    exit_code = 1
    print(f"{FAIL}{msg}")


# 1. Python version
if sys.version_info >= (3, 11):
    vi = sys.version_info
    print(f"{PASS}Python {vi.major}.{vi.minor}.{vi.micro}")
else:
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    fail(f"Python 3.11+ required. You have {v}. Install from https://python.org")

# 2 & 3. Ollama reachability + model check
_ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
try:
    with urllib.request.urlopen(f"{_ollama_host}/api/tags", timeout=3) as resp:
        data = json.loads(resp.read())
    models = [m["name"] for m in data.get("models", [])]
    if models:
        names = ", ".join(models)
        plural = "s" if len(models) != 1 else ""
        print(f"{PASS}Ollama running ({len(models)} model{plural} found: {names})")
    else:
        print(f"{PASS}Ollama running")
        fail("No models found in Ollama.\nFix: ollama pull llama3.1")
except urllib.error.URLError:
    fail("Ollama is not running.\nFix: open a new terminal and run: ollama serve")
except Exception as exc:  # noqa: BLE001
    fail(f"Could not reach Ollama: {exc}\nFix: open a new terminal and run: ollama serve")

# 4. Core dependencies
_core = {
    "langgraph": "langgraph",
    "gradio": "gradio",
    "langchain_ollama": "langchain-ollama",
    "ddgs": "ddgs",
    "sentence_transformers": "sentence-transformers",
}
_core_ok = True
for module, pkg in _core.items():
    try:
        __import__(module)
    except ImportError:
        fail(f"Missing dependency: {pkg}\nFix: uv sync")
        _core_ok = False

if _core_ok:
    print(f"{PASS}Core dependencies installed")

# 5. Optional dependencies
_optional = {
    "faiss": "faiss-cpu or faiss-gpu — URL-based dedup will be used instead of semantic",
    "langchain_anthropic": "langchain-anthropic — Anthropic provider unavailable",
    "langchain_openai": "langchain-openai — OpenAI provider unavailable",
    "tavily": "tavily-python — Tavily search provider unavailable",
}
for module, note in _optional.items():
    try:
        __import__(module)
    except ImportError:
        print(f"{WARN}Optional: {note}")

sys.exit(exit_code)
