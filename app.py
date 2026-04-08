import logging

import gradio as gr
import httpx

from src.agent.graph import build_graph
from src.config import AgentConfig

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
.wrap label span { color: #e8dcc8 !important; font-size: 0.9rem !important; letter-spacing: normal !important; text-transform: none !important; }
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

/* Markdown prose */
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

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #2e2e2e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #c8922a; }
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


def research(query: str, depth: str, model: str):
    """
    Generator — yields (status, report) tuples so Gradio can stream progress.
    Each node completion updates the status bar; the report appears once synthesis finishes.
    """
    query = query.strip()
    if not query:
        yield "Drop a pin on your research topic above.", ""
        return

    cfg = AgentConfig(
        model_name=model,
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

            yield _status_line(node_name, detail, current_retry, cfg.max_retries), report

    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        hint = _ollama_hint(error_msg)
        yield f"Error: {error_msg}{hint}", report
        return

    yield "Your map is ready.", report


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Cartograph", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        "# ◈ Cartograph\n"
        "*Plants a pin. Surveys the terrain. Draws you a map.*"
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(
                label="Territory to Map",
                placeholder="e.g., Latest advances in on-device LLM inference",
                lines=2,
            )
        with gr.Column(scale=1):
            depth_radio = gr.Radio(
                choices=list(DEPTH_MAP.keys()),
                label="Survey Depth",
                value="Standard (5)",
            )

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=_MODEL_CHOICES,
            value=_MODEL_DEFAULT,
            label="Model",
            scale=1,
        )
        run_btn = gr.Button("Chart It", variant="primary", scale=3)

    status_box = gr.Textbox(
        label="Survey Log",
        interactive=False,
        lines=4,
        placeholder="The survey will appear here once you plant a pin…",
    )

    report_box = gr.Markdown(label="Field Report")

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

    run_btn.click(
        fn=research,
        inputs=[query_box, depth_radio, model_dropdown],
        outputs=[status_box, report_box],
    )
    query_box.submit(
        fn=research,
        inputs=[query_box, depth_radio, model_dropdown],
        outputs=[status_box, report_box],
    )

if __name__ == "__main__":
    demo.launch()
