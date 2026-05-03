import logging
import tempfile
from pathlib import Path

import gradio as gr
import httpx

from src.agent.graph import build_graph
from src.config import AgentConfig
from src.history import load_by_id, load_recent, save_run
from src.llm import CLOUD_MODEL_CHOICES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DEPTH_MAP = {
    "Quick (3)": 3,
    "Standard (5)": 5,
    "Deep (7)": 7,
}

NODE_LABELS = {
    "planner": "Charting the route",
    "searcher": "Surveying the terrain",
    "evaluator": "Verifying landmarks",
    "synthesizer": "Drawing the map",
}

EXAMPLES = [
    ["How do mRNA vaccines work and what diseases might they treat next?", "Standard (5)"],
    ["What caused the Bronze Age Collapse around 1200 BCE?", "Quick (3)"],
    ["Economic and social effects of a four-day work week", "Standard (5)"],
    ["State of the art in room-temperature superconductors", "Deep (7)"],
    ["How does the gut microbiome influence mental health?", "Standard (5)"],
]

_STEP_KEYS = list(NODE_LABELS.keys())

# ── Nautical colour palette ────────────────────────────────────────────────────
# Deep navy background, compass-gold accents, aged-parchment text, teal links.

CSS = """
/* Base & container */
body, .gradio-container, .contain, footer {
    background-color: #0d0d0d !important;
}
.gap { background-color: #0d0d0d !important; }

/* Panels */
.block, .form, .box {
    background-color: #1a1a1a !important;
    border-color: #2e2e2e !important;
}

/* Inputs & textareas */
textarea, input[type="text"], input[type="search"] {
    background-color: #111111 !important;
    color: #e8dcc8 !important;
    border-color: #2e2e2e !important;
}
textarea::placeholder { color: #555555 !important; }

/* Radio option text — cream for unselected, black for selected */
.wrap label span {
    color: #e8dcc8 !important; font-size: 0.9rem !important;
    letter-spacing: normal !important; text-transform: none !important;
}
.wrap label.selected span { color: #0d0d0d !important; font-weight: 700 !important; }

/* Subtext under dropdowns */
.info { color: #666666 !important; font-size: 0.75rem !important; }

/* Label badge pills — amber outline, transparent fill */
span[data-testid="block-info"] {
    background: transparent !important;
    border: 1px solid #c8922a !important;
    color: #c8922a !important;
    border-radius: 4px !important;
    padding: 2px 8px !important;
}

/* Radio pills — selected state → amber fill */
.wrap label.selected {
    background: #c8922a !important;
    border-color: #c8922a !important;
    color: #0d0d0d !important;
    font-weight: 700 !important;
}

/* Radio dot — amber accent for unselected, hidden inside selected pill */
input[type="radio"] { accent-color: #c8922a !important; }
label.selected input[type="radio"] { display: none !important; }

/* Primary button — compass gold */
button.primary, button.primary:focus {
    background: linear-gradient(135deg, #c8922a 0%, #a8721a 100%) !important;
    border-color: #a8721a !important;
    color: #0f1923 !important;
    font-weight: 800 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    box-shadow: 0 2px 12px rgba(200, 146, 42, 0.3) !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #d9a23a 0%, #b8821a 100%) !important;
    box-shadow: 0 4px 18px rgba(200, 146, 42, 0.45) !important;
}

/* Secondary buttons */
button.secondary {
    background-color: #1a1a1a !important;
    border-color: #2e2e2e !important;
    color: #e8dcc8 !important;
}

/* Radio & checkbox labels */
.wrap > label, .wrap > label > span { color: #e8dcc8 !important; }
input[type="radio"]:checked + span { color: #0d0d0d !important; font-weight: 700 !important; }

/* Dropdowns */
.wrap select, select {
    background-color: #111111 !important;
    color: #e8dcc8 !important;
    border-color: #2e2e2e !important;
}

/* Markdown prose (title, footer, and other gr.Markdown blocks) */
.prose, .prose p, .prose li { color: #e8dcc8 !important; }
.prose h1, .prose h2, .prose h3, .prose h4 {
    color: #c8922a !important;
    border-bottom-color: #253d52 !important;
}
.prose a { color: #4a9b8e !important; }
.prose a:hover { color: #5ab8aa !important; }
.prose code {
    background-color: #1a2f42 !important;
    color: #c8922a !important;
}

/* Examples table */
.examples { border-color: #2e2e2e !important; }
.examples table { border-color: #2e2e2e !important; border-collapse: collapse !important; }

/* Header row — amber fill, black text, map-legend style */
tr.tr-head { background-color: #c8922a !important; }
tr.tr-head th {
    background-color: transparent !important;
    color: #0d0d0d !important;
    font-weight: 800 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    border-color: #a8721a !important;
    padding: 10px 16px !important;
}

/* Body rows — pure black, no navy */
tr.tr-body {
    background-color: #0d0d0d !important;
    border-bottom: 1px solid #2e2e2e !important;
    transition: background-color 0.15s ease, border-left 0.15s ease !important;
    cursor: pointer !important;
}
tr.tr-body td {
    color: #e8dcc8 !important;
    border-color: #2e2e2e !important;
    padding: 10px 16px !important;
}

/* Hover — amber left-border flash + faint amber tint */
tr.tr-body:hover {
    background-color: rgba(200, 146, 42, 0.08) !important;
    border-left: 3px solid #c8922a !important;
}
tr.tr-body:hover td { color: #c8922a !important; }

/* Controls row — stretch children to equal height */
.controls-row { align-items: stretch !important; }
.controls-row > div { display: flex !important; flex-direction: column !important; }
.chart-btn { flex: 1 !important; }
.chart-btn button { height: 100% !important; }

/* Survey Depth — stretch pills across full row width */
.depth-radio .wrap { display: flex !important; gap: 8px !important; }
.depth-radio .wrap label { flex: 1 !important; justify-content: center !important;
    text-align: center !important; }

/* History accordion */
.history-accordion { margin-top: 12px !important; }
.history-accordion .label-wrap { color: #c8922a !important; font-size: 0.85rem !important; }

/* Report HTML box */
#report-box {
    background-color: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
    color: #e8dcc8 !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}
#report-box h1, #report-box h2, #report-box h3, #report-box h4 {
    color: #c8922a !important;
    margin: 1.2em 0 0.4em !important;
    border-bottom: 1px solid #253d52 !important;
    padding-bottom: 4px !important;
}
#report-box p { margin: 0.6em 0 !important; }
#report-box ul { padding-left: 1.4em !important; margin: 0.4em 0 !important; }
#report-box li { margin: 0.2em 0 !important; }
#report-box a { color: #4a9b8e !important; }
#report-box a:hover { color: #5ab8aa !important; }
#report-box strong { color: #d4bc94 !important; }
#report-box em { color: #b8a880 !important; }

/* Export buttons row */
.export-row { margin-top: 8px !important; }
.export-row button {
    background-color: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    color: #c8922a !important;
    font-size: 0.8rem !important;
    padding: 6px 14px !important;
}
.export-row button:hover {
    border-color: #c8922a !important;
    background-color: rgba(200, 146, 42, 0.08) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #2e2e2e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #c8922a; }

/* Full-width layout + bottom padding so footer is never clipped */
.gradio-container {
    max-width: 100% !important;
    padding-left: 24px !important;
    padding-right: 24px !important;
    padding-bottom: 48px !important;
}

/* Survey Depth pills — never wrap, shrink text on very narrow screens */
.depth-radio .wrap { flex-wrap: nowrap !important; }
@media (max-width: 500px) {
    .depth-radio .wrap label {
        font-size: 0.72rem !important;
        padding: 4px 6px !important;
        min-width: 0 !important;
    }
}

/* report_txt: streaming sink for raw markdown.
   MUST stay in the normal document flow at its natural height so that
   Gradio 6 / Svelte 5 SSE DOM updates reach the textarea.
   opacity: 0 hides it visually while leaving it fully in-flow.
   clip-path breaks SSE delivery in Gradio 6 / Svelte 5. */
#report-txt {
    opacity: 0 !important;
    pointer-events: none !important;
}

"""



def _available_models() -> tuple[list[str], str]:
    """Query Ollama for installed models; return (choices, default)."""
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        preferred = next((m for m in models if m.startswith("llama3.1")), None)
        default = preferred or (models[0] if models else "llama3.1")
        return models or ["llama3.1"], default
    except Exception:  # noqa: BLE001
        return ["llama3.1"], "llama3.1"


_MODEL_CHOICES, _MODEL_DEFAULT = _available_models()

# ── Page-load JS ──────────────────────────────────────────────────────────────
# Two responsibilities:
#   1. Aria-label injection (Gradio doesn't wire these for screen readers).
#   2. Report renderer: polls #report-txt textarea every 200 ms and injects
#      rendered HTML into #report-box whenever the value changes.
#
# Why polling?  Gradio 6 / Svelte 5 streaming updates reliably propagate to
# DOM textareas, but NON-streaming .then() output updates do not (the Svelte
# store is updated but the DOM textarea is never written).  Attempting to
# deliver the report via a .then() chain — whether to gr.HTML, gr.Textbox, or
# a separate "carrier" textbox — all fail at the DOM level.  Streaming,
# however, DOES work: after research completes, #report-txt textarea holds the
# full raw markdown.  Polling that textarea and rendering client-side sidesteps
# the Gradio 6 reactivity gap entirely.
_PAGE_JS = """
function addAriaLabels() {
    const textareas = document.querySelectorAll('textarea');
    if (textareas[0] && !textareas[0].getAttribute('aria-label'))
        textareas[0].setAttribute('aria-label', 'Territory to Map');
    if (textareas[1] && !textareas[1].getAttribute('aria-label'))
        textareas[1].setAttribute('aria-label', 'Survey Log');
    const radios = document.querySelectorAll('.depth-radio input[type="radio"]');
    const depthLabels = ['Quick (3)', 'Standard (5)', 'Deep (7)'];
    radios.forEach((r, i) => {
        if (!r.getAttribute('aria-label') && depthLabels[i])
            r.setAttribute('aria-label', 'Survey Depth: ' + depthLabels[i]);
    });
}

const _cartographObserver = new MutationObserver(() => {
    addAriaLabels();
    const labelled = document.querySelectorAll('.depth-radio input[type="radio"][aria-label]');
    if (labelled.length === 3) _cartographObserver.disconnect();
});
_cartographObserver.observe(document.body, { childList: true, subtree: true });
addAriaLabels();

// ── Report renderer (polling) ─────────────────────────────────────────────────
function _cgEsc(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function _cgRender(mdText) {
    const box = document.getElementById('report-box');
    if (!box) return;
    const ph = '<p><em>Plant a pin above and click <strong>Chart It</strong>'
             + ' — your map will appear here.</em></p>';
    if (!mdText || !mdText.trim()) { box.innerHTML = ph; return; }
    const lines = mdText.split('\\n');
    let html = '', inList = false;
    function closeList() { if(inList){html+='</ul>';inList=false;} }
    for (const line of lines) {
        const e = _cgEsc(line);
        if (/^#### /.test(e))     { closeList(); html+='<h4>'+e.slice(5)+'</h4>'; }
        else if (/^### /.test(e)) { closeList(); html+='<h3>'+e.slice(4)+'</h3>'; }
        else if (/^## /.test(e))  { closeList(); html+='<h2>'+e.slice(3)+'</h2>'; }
        else if (/^# /.test(e))   { closeList(); html+='<h1>'+e.slice(2)+'</h1>'; }
        else if (/^- /.test(e))   {
            if(!inList){html+='<ul>';inList=true;}
            html+='<li>'+e.slice(2)+'</li>';
        }
        else if (!e.trim())       { closeList(); }
        else                      { closeList(); html+='<p>'+e+'</p>'; }
    }
    closeList();
    html = html.replace(/\\*\\*\\*(.+?)\\*\\*\\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\\*\\*(.+?)\\*\\*/g,     '<strong>$1</strong>');
    html = html.replace(/\\*([^*\\n]+?)\\*/g,    '<em>$1</em>');
    // Restrict hrefs to http/https/ftp — prevents javascript: injection from LLM output
    html = html.replace(/\\[([^\\]]+)\\]\\(((?:https?|ftp):\\/\\/[^)]+)\\)/g,
                        '<a href="$2" target="_blank" rel="noopener">$1</a>');
    box.innerHTML = html;
}

let _cgLastTxt = undefined;
const _cgPlaceholderLen = 98; // placeholder innerHTML is ~98 chars
setInterval(function() {
    const el = document.getElementById('report-txt');
    if (!el) return;
    const ta = el.querySelector('textarea');
    if (!ta) return;
    const cur = ta.value;
    const box = document.getElementById('report-box');
    // Re-render if value changed OR if textarea has content but box still shows placeholder
    const boxIsEmpty = !box || box.innerHTML.length <= _cgPlaceholderLen;
    if (cur !== _cgLastTxt || (cur && cur.trim() && boxIsEmpty)) {
        _cgLastTxt = cur;
        _cgRender(cur);
    }
}, 200);
"""


def _status_line(node: str, detail: str = "", retry: int = 0, max_retries: int = 2) -> str:
    label = NODE_LABELS.get(node, node.title())
    step_num = _STEP_KEYS.index(node) + 1 if node in _STEP_KEYS else 1
    total = len(_STEP_KEYS)

    # Route-style progress: ●──◈──○──○
    segments = []
    for i in range(total):
        if i < step_num - 1:
            segments.append("●")
        elif i == step_num - 1:
            segments.append("◈")
        else:
            segments.append("○")
    bar = "──".join(segments)

    retry_tag = f"  [retry {retry}/{max_retries}]" if retry > 0 else ""
    base = f"[{bar}]  Step {step_num}/{total} — {label}…{retry_tag}"
    return f"{base}\n{detail}" if detail else base


def _ollama_hint(error_msg: str) -> str:
    msg = error_msg.lower()
    if "connection" in msg or "refused" in msg:
        return "\n\nHint: Ollama isn't running. Start it with: ollama serve"
    if "not found" in msg or "unknown model" in msg or "pull" in msg:
        return f"\n\nHint: Model not downloaded. Run: ollama pull {_MODEL_DEFAULT}"
    if "timeout" in msg or "timed out" in msg:
        return "\n\nHint: Ollama timed out. The model may be too slow for this hardware."
    return ""


def _history_choices() -> list[tuple[str, int]]:
    return [
        (f"{r['timestamp']} — {r['query'][:60]}", r["id"])
        for r in load_recent(10)
    ]


_REPORT_PLACEHOLDER = (
    "*Plant a pin above and click **Chart It** — your map will appear here.*"
)
_REPORT_PLACEHOLDER_HTML = (
    "<p><em>Plant a pin above and click <strong>Chart It</strong>"
    " — your map will appear here.</em></p>"
)



# ── History helpers ────────────────────────────────────────────────────────────

def _load_history_report(run_id: int | None):
    """Stream raw markdown for the selected history entry → report_txt.

    Yielding rather than returning forces Gradio to use the SSE streaming path,
    which reliably propagates the value to the DOM textarea.  The page-load
    setInterval then detects the change and re-renders #report-box.
    """
    if run_id is None:
        yield ""
        return
    yield load_by_id(run_id) or ""


_last_export_tmp: Path | None = None

# Module-level report cache: the research generator stores the completed report
# here so _after_research can read it without listing report_txt in _after_inputs.
# Listing report_txt in .then() inputs causes Gradio 6 to reset its DOM textarea
# to the initial value ("") after reading it — clearing the streaming-set value
# that the setInterval renderer depends on.
_last_completed_report: str = ""


def _after_research(query: str, depth: str, model: str):
    """Single .then() handler after research completes.

    Returns 3 values: [history_dropdown, export_row, dl_btn].
    Reads the completed report from _last_completed_report (set by the research
    generator) rather than from a Gradio input component — that avoids the
    Gradio 6 behaviour where listing report_txt in .then() inputs resets its
    DOM textarea, erasing the streaming-written value the setInterval needs.
    """
    global _last_export_tmp
    report = _last_completed_report

    _good = (
        bool(report)
        and not report.startswith("Error:")
        and not report.startswith("Drop a pin")
        and report != _REPORT_PLACEHOLDER
    )

    # ── history ───────────────────────────────────────────────────────────────
    if _good:
        save_run(query, depth, model, report)

    # ── export ────────────────────────────────────────────────────────────────
    if _good:
        if _last_export_tmp is not None:
            _last_export_tmp.unlink(missing_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".md", mode="w", encoding="utf-8"
        )
        tmp.write(report)
        tmp.close()
        _last_export_tmp = Path(tmp.name)
        return (
            gr.update(choices=_history_choices(), value=None),
            gr.update(visible=True),
            gr.update(value=tmp.name, visible=True),
        )

    return (
        gr.update(choices=_history_choices(), value=None),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
    )




def research(
    query: str, depth: str, model: str, cloud_model: str,
    provider: str, api_key: str,
):
    """Generator — yields (status, raw_markdown) pairs to [status_box, report_txt].

    report_txt must be interactive=False and in-flow (not off-screen absolute).
    Stores the completed report in _last_completed_report so _after_research
    can read it without listing report_txt in .then() inputs.
    """
    global _last_completed_report
    query = query.strip()
    if not query:
        yield "Drop a pin on your research topic above.", ""
        return
    if len(query) < 10:
        yield "Your query is too short — try describing your topic in more detail.", ""
        return
    if len(query) > 500:
        yield "Query too long (500 character max) — try a more focused topic.", ""
        return

    is_ollama = provider == "Ollama (local)"
    cfg = AgentConfig(
        provider="ollama" if is_ollama else provider.lower(),
        api_key="" if is_ollama else (api_key or "").strip(),
        model_name=model if is_ollama else cloud_model,
        max_sub_questions=DEPTH_MAP[depth],
    )
    graph = build_graph(cfg)

    initial_state = {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "evaluated_sources": [],
        "coverage_sufficient": False,
        "retry_count": 0,
        "report": "",
    }

    report = ""
    current_retry = 0

    try:
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            node_name = next(iter(chunk))
            node_output = chunk[node_name]
            detail = ""

            if node_name == "planner":
                sqs = node_output.get("sub_questions", [])
                if sqs:
                    detail = "Sub-questions:\n" + "\n".join(f"  • {q}" for q in sqs)

            elif node_name == "searcher":
                n = len(node_output.get("search_results", []))
                detail = f"{n} sources surveyed"

            elif node_name == "evaluator":
                evaluated = node_output.get("evaluated_sources", [])
                current_retry = max(0, node_output.get("retry_count", 1) - 1)
                detail = f"{len(evaluated)} landmarks verified"
                if not node_output.get("coverage_sufficient"):
                    detail += " — coverage thin, extending survey"

            elif node_name == "synthesizer":
                report = node_output.get("report", "")

            status = _status_line(node_name, detail, current_retry, cfg.max_retries)
            yield status, report

    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        hint = _ollama_hint(error_msg)
        _last_completed_report = ""
        yield f"Error: {error_msg}{hint}", ""
        return

    _last_completed_report = report
    yield "Your map is ready.", report


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Cartograph") as demo:
    gr.Markdown(
        "# ◈ Cartograph\n"
        "*Plants a pin. Surveys the terrain. Draws you a map.*"
    )

    query_box = gr.Textbox(
        label="Territory to Map",
        placeholder="e.g., Latest advances in on-device LLM inference",
        lines=1,
    )

    depth_radio = gr.Radio(
        choices=list(DEPTH_MAP.keys()),
        label="Survey Depth",
        value="Standard (5)",
        elem_classes="depth-radio",
    )

    provider_radio = gr.Radio(
        choices=["Ollama (local)", "Anthropic", "OpenAI"],
        value="Ollama (local)",
        label="Provider",
        elem_classes="depth-radio",
    )

    with gr.Row(elem_classes="controls-row"):
        model_dropdown = gr.Dropdown(
            choices=_MODEL_CHOICES,
            value=_MODEL_DEFAULT,
            label="Model",
            scale=2,
            visible=True,
        )
        cloud_model_dropdown = gr.Dropdown(
            choices=CLOUD_MODEL_CHOICES["Anthropic"],
            value=CLOUD_MODEL_CHOICES["Anthropic"][0],
            label="Model",
            scale=2,
            visible=False,
        )
        api_key_box = gr.Textbox(
            label="API Key",
            placeholder="sk-ant-... or sk-...",
            type="password",
            lines=1,
            scale=2,
            visible=False,
        )
        run_btn = gr.Button("Chart It", variant="primary", scale=3, elem_classes="chart-btn")

    status_box = gr.Textbox(
        label="Survey Log",
        interactive=False,
        lines=2,  # keep page height ≤ viewport so controls stay visible after example click
        placeholder="The survey will appear here once you plant a pin…",
    )

    # report_box uses gr.HTML so it can be updated via .then() after research.
    # gr.Markdown does not respond to Gradio 6 update events (Svelte 5 reactivity
    # gap); gr.HTML's component handles value injection correctly.
    report_box = gr.HTML(
        value=_REPORT_PLACEHOLDER_HTML, label="Field Report", elem_id="report-box"
    )

    # report_txt: streaming sink for raw markdown.
    # Must be interactive=False (Gradio 6 / Svelte 5 uses bind:value for
    # interactive textboxes, which blocks SSE updates reaching the DOM).
    # Must NOT use off-screen CSS (position:absolute; left:-9999px) — that
    # removes it from the document flow and Gradio's Svelte runtime then skips
    # DOM updates for it.  Hidden via opacity:0 in CSS so it stays in-flow.
    report_txt = gr.Textbox(
        value="", interactive=False, visible=True,
        elem_id="report-txt", label="", lines=1,
    )

    with gr.Row(visible=False, elem_classes="export-row") as export_row:
        copy_btn = gr.Button("⎘ Copy Markdown", variant="secondary", scale=1)
        dl_btn = gr.DownloadButton("↓ Download .md", variant="secondary", scale=1, visible=False)

    with gr.Accordion("Recent Maps", open=False, elem_classes="history-accordion"):
        history_dropdown = gr.Dropdown(
            choices=_history_choices(),
            label="Select a past report to reload",
            value=None,
            interactive=True,
        )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[query_box, depth_radio],
        label="Uncharted territories — click to explore",
    )

    gr.Markdown(
        "<div style='text-align:center; color:#444444; font-size:0.75rem; padding:8px 0 4px;'>"
        "Powered by LangGraph"
        "</div>"
    )

    def _on_provider_change(provider: str):
        is_ollama = provider == "Ollama (local)"
        cloud_choices = CLOUD_MODEL_CHOICES.get(provider, CLOUD_MODEL_CHOICES["Anthropic"])
        return (
            gr.update(visible=is_ollama),
            gr.update(visible=not is_ollama, choices=cloud_choices, value=cloud_choices[0]),
            gr.update(visible=not is_ollama),
        )

    provider_radio.change(
        fn=_on_provider_change,
        inputs=[provider_radio],
        outputs=[model_dropdown, cloud_model_dropdown, api_key_box],
    )

    _inputs = [
        query_box, depth_radio, model_dropdown, cloud_model_dropdown,
        provider_radio, api_key_box,
    ]
    # _after_research reads the report from _last_completed_report (module-level
    # cache) so report_txt is NOT listed in _after_inputs.  Listing report_txt
    # in .then() inputs would cause Gradio 6 to reset its DOM textarea to ""
    # after reading — erasing the streaming-written value the setInterval needs.
    _after_inputs = [query_box, depth_radio, model_dropdown]
    _after_outputs = [history_dropdown, export_row, dl_btn]

    # Force-render JS: runs after _after_research completes, guaranteeing the
    # SSE-delivered report in #report-txt textarea is rendered into #report-box.
    # The setInterval polling handles intermediate streaming renders; this .then()
    # is the reliable final render once the full run has settled.
    _force_render_js = (
        "() => { const ta = document.getElementById('report-txt')"
        "?.querySelector('textarea');"
        " if (ta?.value?.trim()) window._cgRender(ta.value); return []; }"
    )

    run_btn.click(
        fn=research, inputs=_inputs, outputs=[status_box, report_txt],
    ).then(
        fn=_after_research, inputs=_after_inputs, outputs=_after_outputs,
    ).then(fn=None, js=_force_render_js)

    query_box.submit(
        fn=research, inputs=_inputs, outputs=[status_box, report_txt],
    ).then(
        fn=_after_research, inputs=_after_inputs, outputs=_after_outputs,
    ).then(fn=None, js=_force_render_js)

    # History load: yield once → SSE streaming path → report_txt DOM updated →
    # setInterval detects change → re-renders #report-box.
    history_dropdown.change(
        fn=_load_history_report,
        inputs=[history_dropdown],
        outputs=[report_txt],
    )

    # Copy: read from report_txt DOM textarea (populated by streaming).
    copy_btn.click(
        fn=None,
        js="(report) => { navigator.clipboard.writeText(report); return []; }",
        inputs=[report_txt],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS, js=_PAGE_JS)
