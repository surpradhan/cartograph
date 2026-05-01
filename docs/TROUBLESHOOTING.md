# Cartograph — Troubleshooting

---

### 1. `Connection refused` or `Failed to connect to Ollama`

**Symptom:** Error when starting the app or during a research run.  
**Cause:** The Ollama process is not running.  
**Fix:**
```bash
ollama serve        # start Ollama in a separate terminal
# then rerun: python app.py
```

---

### 2. `unknown model` or `model not found`

**Symptom:** Error during the planning or synthesis step.  
**Cause:** The selected model hasn't been pulled to local Ollama.  
**Fix:**
```bash
ollama pull llama3.1          # or whichever model you selected
ollama list                   # verify it appears
```

---

### 3. `Port 7860 is already in use`

**Symptom:** App fails to start.  
**Cause:** Another Gradio app or service is using port 7860.  
**Fix:** Set a different port in `.env`:
```
GRADIO_PORT=7861
```
Then restart `python app.py`.

---

### 4. `ModuleNotFoundError` for any package

**Symptom:** Import error on startup.  
**Cause:** Virtual environment not activated or `uv sync` not run.  
**Fix:**
```bash
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv sync
python app.py
```

---

### 5. LLM response is very slow or times out

**Symptom:** Research hangs at "Charting the route" or times out after 120s.  
**Cause:** Llama 3.1 8B is too large for available RAM/VRAM.  
**Fix options:**
- Use a smaller model: `ollama pull phi3` or `ollama pull mistral`
- Select the smaller model from the Model dropdown in the UI
- Increase the timeout in `src/config.py`: `llm_timeout = 300`

---

### 6. DuckDuckGo returns no results or rate-limit error

**Symptom:** "0 sources surveyed" or `DDGSException` in logs.  
**Cause:** DuckDuckGo rate-limited the request (common with quick repeated runs).  
**Fix:** Wait 30–60 seconds before retrying. For sustained use, switch to Tavily:
```bash
uv add tavily-python
# Get a free API key at https://tavily.com
# Enter it in the Tavily API Key field in the UI
```

---

### 7. `langchain_anthropic` or `langchain_openai` not found

**Symptom:** Error when switching Provider to Anthropic or OpenAI in the UI.  
**Cause:** Cloud provider extras are not installed by default.  
**Fix:**
```bash
uv add langchain-anthropic   # for Anthropic
uv add langchain-openai      # for OpenAI
```
