"""
PromptLab Pro v3 — app.py
══════════════════════════
Research-grade Prompt Evaluation Platform
Powered by OpenRouter | Built with Streamlit
"""
from __future__ import annotations

import streamlit as st
import os, json, time
from datetime import datetime
from dotenv import load_dotenv

from modules.llm_client  import get_client, call_llm, APP_VERSION
from modules.analysis    import (
    pairwise_similarity, length_stats, determinism_index,
    detect_outliers, hallucination_risk, validate_json,
    instruction_adherence, word_diff,
)
from modules.ai_summary  import generate_summary
from modules.export      import (
    save_config, load_config, to_json, to_csv, to_markdown,
    build_experiment_dict,
)
from modules.templates   import TEMPLATES

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PromptLab Pro",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg:       #060810;
    --surf:     #0a0e1a;
    --card:     #0e1422;
    --border:   #162040;
    --border-h: #1e2f55;
    --a:        #60a5fa;
    --a-glow:   rgba(96,165,250,0.15);
    --a-glow2:  rgba(96,165,250,0.06);
    --b:        #a78bfa;
    --b-glow:   rgba(167,139,250,0.15);
    --b-glow2:  rgba(167,139,250,0.06);
    --g:        #34d399;
    --y:        #fbbf24;
    --text:     #e2e8f0;
    --text-2:   #94a3b8;
    --muted:    #475569;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(96,165,250,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(167,139,250,0.07) 0%, transparent 60%),
        var(--bg) !important;
}

/* ── Animated noise grain overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── Header ── */
.plp-header {
    padding: 2.5rem 0 2rem;
    margin-bottom: 2rem;
    position: relative;
}
.plp-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #60a5fa, #a78bfa, transparent);
    opacity: 0.4;
}
.plp-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--a);
    opacity: 0.7;
    margin-bottom: 0.6rem;
}
.plp-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -1.5px;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.plp-title span {
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.plp-desc {
    font-size: 0.82rem;
    color: var(--text-2);
    letter-spacing: 0.3px;
    margin-top: 0.3rem;
}

/* ── Live stats bar ── */
.stats-bar {
    display: flex;
    gap: 2rem;
    padding: 1rem 1.5rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.stats-bar::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #60a5fa, transparent);
    opacity: 0.5;
}
.stat-item { text-align: center; flex: 1; }
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--a);
    line-height: 1;
    display: block;
}
.stat-val.pink { color: var(--b); }
.stat-val.green { color: var(--g); }
.stat-lbl {
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.3rem;
    display: block;
}
.stat-divider {
    width: 1px;
    background: var(--border);
    align-self: stretch;
}

/* ── Section labels ── */
.sec {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Glowing cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.card:hover {
    border-color: var(--border-h);
    box-shadow: 0 0 30px rgba(96,165,250,0.04);
}
.card.glow-a {
    border-color: rgba(96,165,250,0.25);
    box-shadow: 0 0 40px rgba(96,165,250,0.07), inset 0 1px 0 rgba(96,165,250,0.1);
}
.card.glow-b {
    border-color: rgba(167,139,250,0.25);
    box-shadow: 0 0 40px rgba(167,139,250,0.07), inset 0 1px 0 rgba(167,139,250,0.1);
}

/* ── Response cards ── */
.resp {
    background: var(--surf);
    border: 1px solid var(--border);
    border-left: 2px solid var(--a);
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    border-radius: 0 10px 10px 0;
    font-size: 0.81rem;
    line-height: 1.85;
    white-space: pre-wrap;
    word-break: break-word;
    transition: border-color 0.2s, box-shadow 0.2s;
    animation: slideIn 0.3s ease;
}
.resp:hover {
    border-left-color: var(--a);
    box-shadow: -4px 0 20px rgba(96,165,250,0.15), 0 4px 20px rgba(0,0,0,0.3);
}
.resp.b {
    border-left-color: var(--b);
}
.resp.b:hover {
    border-left-color: var(--b);
    box-shadow: -4px 0 20px rgba(167,139,250,0.15), 0 4px 20px rgba(0,0,0,0.3);
}
.resp.err { border-left-color: #ef4444; opacity: .7; }
.resp.outlier {
    border-left-color: var(--y);
    background: rgba(251,191,36,0.03);
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-8px); }
    to   { opacity: 1; transform: translateX(0); }
}

.run-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.run-lbl::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    opacity: 0.5;
}

/* ── Metric boxes ── */
.mbox {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
}
.mbox::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, currentColor, transparent);
    opacity: 0.3;
}
.mbox:hover {
    border-color: var(--border-h);
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.mval {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -1px;
}
.mlbl {
    font-size: 0.58rem;
    letter-spacing: 2px;
    color: var(--muted);
    margin-top: 0.35rem;
    text-transform: uppercase;
}

/* ── Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-weight: 600;
    margin-right: 0.3rem;
    letter-spacing: 0.5px;
}
.bg { background: rgba(52,211,153,.1);  color: var(--g); border: 1px solid rgba(52,211,153,.25); }
.by { background: rgba(251,191,36,.1);  color: var(--y); border: 1px solid rgba(251,191,36,.25); }
.br { background: rgba(244,114,182,.1); color: var(--b); border: 1px solid rgba(244,114,182,.25); }
.ba { background: var(--a-glow);        color: var(--a); border: 1px solid rgba(56,189,248,.25); }

/* ── Diff ── */
.dadd { background: rgba(52,211,153,.12); color: var(--g); padding: 1px 4px; border-radius: 3px; }
.drem { background: rgba(244,114,182,.12); color: var(--b); padding: 1px 4px; border-radius: 3px; text-decoration: line-through; }

/* ── Meta tags ── */
.mt {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text-2);
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    letter-spacing: 0.3px;
}

/* ── Why box ── */
.why-box {
    background: rgba(96,165,250,0.05);
    border: 1px solid rgba(96,165,250,0.15);
    border-radius: 12px;
    padding: 1.4rem;
    font-size: 0.82rem;
    line-height: 1.8;
    color: var(--text-2);
}
.why-box ul { padding-left: 1.2rem; margin: 0.5rem 0; }
.why-box li { margin-bottom: 0.4rem; }

/* ── Animated run button ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #8b5cf6 100%) !important;
    background-size: 200% 200% !important;
    color: #060810 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    animation: gradShift 4s ease infinite !important;
    box-shadow: 0 4px 20px rgba(96,165,250,0.3) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(96,165,250,0.35) !important;
}
@keyframes gradShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    border-radius: 4px !important;
}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--a) !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,0.2) !important;
    outline: none !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080c18 !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] > div {
    background: var(--surf) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: 8px !important;
}

/* ── Labels ── */
label {
    color: var(--muted) !important;
    font-size: 0.62rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 1.1rem !important;
    transition: color 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--a) !important;
    border-bottom: 2px solid var(--a) !important;
}

/* ── Slider thumb ── */
.stSlider [data-baseweb="thumb"] {
    background: var(--a) !important;
    box-shadow: 0 0 10px rgba(59,130,246,0.3) !important;
}
.stSlider [data-baseweb="track-fill"] {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
}

/* ── Toggle ── */
.stCheckbox > label, .st-emotion-cache-1kyxreq {
    color: var(--text-2) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-h); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="plp-header">
    <p class="plp-eyebrow">⚗️ Prompt Evaluation Platform</p>
    <h1 class="plp-title">PromptLab <span>Pro</span></h1>
    <p class="plp-desc">Measure consistency, detect hallucination risk, and compare prompts scientifically — v{APP_VERSION}</p>
</div>
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-val">3</span>
        <span class="stat-lbl">Experiment Modes</span>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
        <span class="stat-val pink">5</span>
        <span class="stat-lbl">Free Models</span>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
        <span class="stat-val green">DI</span>
        <span class="stat-lbl">Determinism Index</span>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
        <span class="stat-val">JSON</span>
        <span class="stat-lbl">Export Ready</span>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
        <span class="stat-val pink">AI</span>
        <span class="stat-lbl">Auto Summary</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sec">🎛️ Configuration</p>', unsafe_allow_html=True)

    model = st.selectbox("Model", [
        "openrouter/free",
        "qwen/qwen-2.5-next-80b-a3b-instruct:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "openai/gpt-oss-120b:free",
        "minimax/minimax-m2.5:free",
    ])

    mode = st.radio("Experiment Mode", [
        "🔁 Multi-Run Variability",
        "🌡️ Temperature Sweep",
        "🆚 Prompt Comparison",
    ])

    st.markdown("---")
    st.markdown('<p class="sec">⚙️ Parameters</p>', unsafe_allow_html=True)

    if mode == "🌡️ Temperature Sweep":
        temps_str = st.text_input("Temperatures (comma-separated)", "0.0, 0.3, 0.7, 1.0, 1.5")
        try:
            temperatures = [float(x.strip()) for x in temps_str.split(",")]
        except:
            temperatures = [0.0, 0.7, 1.0]
        temperature = temperatures[0]
        n_runs = 1
    else:
        temperature  = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        n_runs       = st.select_slider("Runs", [2, 3, 4, 5], value=3)
        temperatures = [temperature]

    max_tokens = st.slider("Max Tokens", 128, 1024, 512, 64)

    st.markdown("---")
    st.markdown('<p class="sec">🔧 Options</p>', unsafe_allow_html=True)
    json_mode         = st.toggle("Force JSON Output",          value=False)
    show_hallucination = st.toggle("Hallucination Heuristic",   value=True)
    show_adherence    = st.toggle("Instruction Adherence",       value=True)
    show_ai_summary   = st.toggle("AI-Generated Summary",        value=True)
    show_diff         = st.toggle("Word Diff Viewer",            value=True)
    outlier_thresh    = st.slider("Outlier Threshold (%)", 20, 80, 50)

    st.markdown("---")
    st.markdown('<p class="sec">📚 Templates</p>', unsafe_allow_html=True)
    template_choice = st.selectbox("Load Template", ["— Custom —"] + list(TEMPLATES.keys()))

    st.markdown("---")
    st.markdown('<p class="sec">🔄 Reproducibility</p>', unsafe_allow_html=True)
    uploaded_config = st.file_uploader("Upload Config JSON", type=["json"])


# ── Load template / config ────────────────────────────────────────────────────
tpl = TEMPLATES.get(template_choice) if template_choice != "— Custom —" else None
cfg = None
if uploaded_config:
    try:
        cfg = load_config(uploaded_config.read().decode())
        st.sidebar.success("✅ Config loaded — fields pre-filled")
    except ValueError as e:
        st.sidebar.error(f"Config error: {e}")


# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec">📝 Inputs</p>', unsafe_allow_html=True)

task = st.text_area(
    "Task / Question",
    value=cfg.get("task", tpl["example_task"] if tpl else "") if cfg else (tpl["example_task"] if tpl else ""),
    placeholder="e.g. What is recursion in programming?",
    height=68,
)

if mode in ("🔁 Multi-Run Variability", "🌡️ Temperature Sweep"):
    prompt_a = st.text_area(
        "Prompt Template  (use {task} as placeholder)",
        value=cfg.get("prompt_a", tpl["template_a"] if tpl else "") if cfg else (tpl["template_a"] if tpl else ""),
        placeholder="Explain {task} step by step.",
        height=110,
    )
    prompt_b = ""
else:
    ca, cb = st.columns(2, gap="medium")
    with ca:
        prompt_a = st.text_area("Prompt A",
            value=cfg.get("prompt_a", tpl["template_a"] if tpl else "") if cfg else (tpl["template_a"] if tpl else ""),
            placeholder="You are a teacher. Explain {task} step by step.", height=110)
    with cb:
        prompt_b = st.text_area("Prompt B",
            value=cfg.get("prompt_b", tpl["template_b"] if tpl else "") if cfg else (tpl["template_b"] if tpl else ""),
            placeholder="Explain {task} in one sentence.", height=110)

run_btn = st.button("⚗️  Run Experiment", use_container_width=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_prompt(template: str, t: str) -> str:
    return template.replace("{task}", t) if "{task}" in template else f"{template}\n\nTask: {t}"

def render_resp(r: dict, idx: int, css: str = "a", label_override: str = ""):
    lbl = label_override or f"RUN {idx+1}"
    meta = f"{len(r['text'].split())} words · {r['latency_ms']}ms"
    if r.get("tokens_completion"):
        meta += f" · {r['tokens_completion']} tok"
    if r.get("error"):
        st.markdown(f'<div class="resp err"><div class="run-lbl">{lbl} — ERROR</div>{r["error"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="resp {css}"><div class="run-lbl">{lbl} &nbsp;|&nbsp; {meta}</div>{r["text"]}</div>', unsafe_allow_html=True)

def metric_row(vals: list[tuple[str, str, str]]):
    cols = st.columns(len(vals))
    for col, (val, lbl, color) in zip(cols, vals):
        with col:
            st.markdown(f'<div class="mbox"><div class="mval" style="color:{color}">{val}</div><div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

def badge(text: str, score_or_level: str) -> str:
    cls = "bg" if any(x in score_or_level for x in ["Low","High Cons","Highly S","🟢"]) \
        else ("by" if any(x in score_or_level for x in ["Med","Mod","🟡"]) else "br")
    return f'<span class="badge {cls}">{text}</span>'

def run_batch(client, prompt_tmpl, task_txt, model, temp, n, max_tok, json_m, pb, label):
    full = build_prompt(prompt_tmpl, task_txt)
    results = []
    for i in range(n):
        pb.progress((i+.5)/n, text=f"{label} — run {i+1}/{n}")
        results.append(call_llm(client, full, model, temp, max_tok, json_m))
        time.sleep(0.1)
    pb.progress(1.0, text=f"{label} — done ✓")
    return results


# ── WHY THIS MATTERS ─────────────────────────────────────────────────────────
with st.expander("ℹ️ Why systematic prompt evaluation matters"):
    st.markdown("""
<div class="why-box">
LLMs are probabilistic systems — the same prompt at temperature > 0 will produce different outputs on every call.
This variability is not a bug; it is fundamental to how these models work.
<br><br>
For production systems, this creates real engineering challenges:
<ul>
<li>A chatbot that gives inconsistent answers erodes user trust</li>
<li>An extraction pipeline that occasionally produces malformed JSON will silently corrupt data</li>
<li>A summarization model that hallucinates statistics can cause downstream decisions to fail</li>
</ul>
Systematic evaluation — running prompts multiple times, measuring consistency, checking output structure, and flagging risk — is not optional for serious deployment. It is the minimum due diligence before shipping an LLM-powered feature.
<br><br>
This tool operationalises that evaluation workflow.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    if not task.strip():
        st.warning("⚠️ Enter a task."); st.stop()
    if not prompt_a.strip():
        st.warning("⚠️ Enter Prompt A."); st.stop()
    if mode == "🆚 Prompt Comparison" and not prompt_b.strip():
        st.warning("⚠️ Enter Prompt B."); st.stop()

    client = get_client()
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Metadata strip
    st.markdown("---")
    st.markdown(
        f'<span class="mt">🤖 {model}</span>'
        f'<span class="mt">🌡️ {temperature}</span>'
        f'<span class="mt">🔁 {n_runs} runs</span>'
        f'<span class="mt">📅 {ts}</span>'
        f'<span class="mt">📦 v{APP_VERSION}</span>'
        + (f'<span class="mt">📋 JSON</span>' if json_mode else ""),
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── MULTI-RUN ──────────────────────────────────────────────────────────
    if mode == "🔁 Multi-Run Variability":
        pb = st.progress(0)
        results = run_batch(client, prompt_a, task, model, temperature, n_runs, max_tokens, json_mode, pb, "Running")
        texts   = [r["text"] for r in results if not r.get("error")]

        sim  = pairwise_similarity(texts) if texts else {"avg_sim": 0, "pairs": [], "matrix": []}
        lsta = length_stats(texts) if texts else {}
        di   = determinism_index(sim["avg_sim"], lsta.get("cv", 0)) if texts else {"di": 0, "label": "N/A", "interpretation": ""}
        outs = detect_outliers(texts, outlier_thresh)

        tab_raw, tab_met, tab_ana, tab_exp = st.tabs([
            "📋 Raw Outputs", "📊 Metrics", "🔬 Analysis", "📤 Export"
        ])

        with tab_raw:
            st.markdown('<p class="sec">Outputs</p>', unsafe_allow_html=True)
            for i, r in enumerate(results):
                extra_css = " outlier" if i in outs else ""
                lbl = f"RUN {i+1}" + (" ⚠️ OUTLIER" if i in outs else "")
                if r.get("error"):
                    st.markdown(f'<div class="resp err"><div class="run-lbl">{lbl}</div>{r["error"]}</div>', unsafe_allow_html=True)
                else:
                    meta = f"{len(r['text'].split())} words · {r['latency_ms']}ms"
                    st.markdown(f'<div class="resp a{extra_css}"><div class="run-lbl">{lbl} &nbsp;|&nbsp; {meta}</div>{r["text"]}</div>', unsafe_allow_html=True)
                if json_mode and not r.get("error"):
                    jv = validate_json(r["text"])
                    cls = "bg" if jv["valid"] else "br"
                    txt = "✓ Valid JSON" if jv["valid"] else f"✗ {jv['error']}"
                    st.markdown(f'<span class="badge {cls}">{txt}</span>', unsafe_allow_html=True)

        with tab_met:
            if texts:
                metric_row([
                    (str(sim["avg_sim"])+"%", "Avg Similarity",    "#38bdf8"),
                    (str(di["di"]),           "Determinism Index", "#38bdf8"),
                    (str(lsta["mean"]),        "Avg Words",         "#38bdf8"),
                    (str(lsta["std_dev"]),     "Std Dev",           "#38bdf8"),
                    (str(lsta["range"]),       "Length Range",      "#38bdf8"),
                ])
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="card">{di["interpretation"]}<br><br><b>Stability:</b> {di["label"]}</div>', unsafe_allow_html=True)

                if outs:
                    st.warning(f"⚠️ Outlier runs detected: {[f'Run {i+1}' for i in outs]} — similarity below {outlier_thresh}%")

                # Length bar chart
                import pandas as pd
                df = pd.DataFrame({"Words": lsta["lengths"]}, index=[f"Run {i+1}" for i in range(len(lsta["lengths"]))])
                st.bar_chart(df, color="#38bdf8", height=160)

        with tab_ana:
            if texts:
                # Hallucination
                if show_hallucination:
                    st.markdown('<p class="sec">⚠️ Hallucination Risk</p>', unsafe_allow_html=True)
                    for i, t in enumerate(texts):
                        hr = hallucination_risk(t)
                        cls = "bg" if "LOW" in hr["level"] else ("by" if "MEDIUM" in hr["level"] else "br")
                        st.markdown(f'**Run {i+1}:** <span class="badge {cls}">{hr["level"]}</span> score={hr["score"]}', unsafe_allow_html=True)
                        if hr["flags"]:
                            with st.expander(f"Run {i+1} flags ({len(hr['flags'])})"):
                                for f in hr["flags"]:
                                    st.markdown(f"- {f}")

                # Instruction adherence
                if show_adherence:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="sec">📐 Instruction Adherence</p>', unsafe_allow_html=True)
                    for i, t in enumerate(texts):
                        ia = instruction_adherence(prompt_a, t)
                        color = "#10b981" if ia["score"] >= 75 else ("#f59e0b" if ia["score"] >= 50 else "#ef4444")
                        st.markdown(f'**Run {i+1}:** <span style="color:{color};font-weight:700">{ia["score"]}/100</span>', unsafe_allow_html=True)
                        for chk in ia["checks"]:
                            icon = "✅" if chk["passed"] else "❌"
                            st.markdown(f"&nbsp;&nbsp;{icon} {chk['rule']}")

                # Diff
                if show_diff and len(texts) >= 2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="sec">🔍 Word Diff — Run 1 vs Run 2</p>', unsafe_allow_html=True)
                    diff = word_diff(texts[0], texts[1])
                    html = " ".join(
                        f'<span class="dadd">{c["text"]}</span>' if c["type"]=="added"
                        else f'<span class="drem">{c["text"]}</span>' if c["type"]=="removed"
                        else c["text"]
                        for c in diff
                    )
                    st.markdown(f'<div class="card" style="font-size:.8rem;line-height:1.9">{html}</div>', unsafe_allow_html=True)

                # AI Summary
                if show_ai_summary:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="sec">🧠 AI-Generated Analytical Summary</p>', unsafe_allow_html=True)
                    with st.spinner("Generating research note…"):
                        summary = generate_summary(
                            client=client, model=model, mode=mode, task=task,
                            prompt_a=prompt_a, prompt_b=None,
                            texts_a=texts, texts_b=[],
                            metrics_a=lsta, metrics_b={},
                            di_a=di, di_b={},
                            temperatures=temperatures,
                        )
                    st.markdown(f'<div class="card" style="font-size:.82rem;line-height:1.85">{summary}</div>', unsafe_allow_html=True)

        with tab_exp:
            _exp_dict = build_experiment_dict(
                prompt_a=prompt_a, prompt_b="", task=task, model=model,
                temperature=temperature, n_runs=n_runs,
                responses_a=results, responses_b=[],
                metrics_a=lsta, metrics_b={},
                di_a=di, di_b={},
                ai_summary=summary if show_ai_summary and texts else "",
                mode=mode,
            )
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.download_button("⬇️ JSON", to_json(_exp_dict), f"pl_{_ts}.json", "application/json", use_container_width=True)
            with c2:
                st.download_button("⬇️ CSV",  to_csv(_exp_dict),  f"pl_{_ts}.csv",  "text/csv", use_container_width=True)
            with c3:
                st.download_button("⬇️ Markdown", to_markdown(_exp_dict), f"pl_{_ts}.md", "text/markdown", use_container_width=True)
            with c4:
                cfg_json = save_config(
                    prompt_a=prompt_a, prompt_b="", task=task, model=model,
                    temperature=temperature, n_runs=n_runs, max_tokens=max_tokens,
                    temperatures=temperatures, json_mode=json_mode, mode=mode,
                )
                st.download_button("💾 Save Config", cfg_json, f"config_{_ts}.json", "application/json", use_container_width=True)


    # ── TEMPERATURE SWEEP ──────────────────────────────────────────────────
    elif mode == "🌡️ Temperature Sweep":
        full = build_prompt(prompt_a, task)
        sweep: dict[float, dict] = {}
        pb = st.progress(0)
        for ti, temp in enumerate(temperatures):
            pb.progress((ti+.5)/len(temperatures), text=f"temp={temp}…")
            sweep[temp] = call_llm(client, full, model, temp, max_tokens, json_mode)
            time.sleep(0.1)
        pb.progress(1.0, text="Sweep complete ✓")

        tab_raw, tab_ana, tab_exp = st.tabs(["📋 Outputs by Temperature", "📊 Analysis", "📤 Export"])

        with tab_raw:
            for temp, r in sweep.items():
                color = "#3b82f6" if temp <= 0.3 else ("#f59e0b" if temp <= 1.0 else "#ef4444")
                tone  = "deterministic" if temp == 0 else ("balanced" if temp <= 0.7 else "creative/chaotic")
                c_tmp, c_res = st.columns([1, 5], gap="medium")
                with c_tmp:
                    st.markdown(f'<div class="mbox" style="margin-top:.5rem"><div class="mval" style="color:{color}">{temp}</div><div class="mlbl">temp</div><div style="color:{color};font-size:.6rem;margin-top:.25rem">{tone}</div></div>', unsafe_allow_html=True)
                with c_res:
                    render_resp(r, 0, "a", label_override=f"TEMP {temp}")

        with tab_ana:
            valid = {t: r["text"] for t, r in sweep.items() if not r.get("error") and r.get("text")}
            if len(valid) >= 2:
                import pandas as pd
                lengths = {str(t): len(txt.split()) for t, txt in valid.items()}
                df = pd.DataFrame({"Words": list(lengths.values())}, index=list(lengths.keys()))
                st.markdown('<p class="sec">Response Length vs Temperature</p>', unsafe_allow_html=True)
                st.bar_chart(df, color="#38bdf8", height=180)

                if show_hallucination:
                    st.markdown('<p class="sec">Hallucination Risk vs Temperature</p>', unsafe_allow_html=True)
                    for temp, txt in valid.items():
                        hr = hallucination_risk(txt)
                        cls = "bg" if "LOW" in hr["level"] else ("by" if "MEDIUM" in hr["level"] else "br")
                        st.markdown(f'**temp={temp}:** <span class="badge {cls}">{hr["level"]}</span>', unsafe_allow_html=True)

                if show_ai_summary:
                    st.markdown('<p class="sec">🧠 AI Summary</p>', unsafe_allow_html=True)
                    with st.spinner("Generating…"):
                        summary = generate_summary(
                            client=client, model=model, mode=mode, task=task,
                            prompt_a=prompt_a, prompt_b=None,
                            texts_a=list(valid.values()), texts_b=[],
                            metrics_a=length_stats(list(valid.values())), metrics_b={},
                            di_a={}, di_b={}, temperatures=temperatures,
                        )
                    st.markdown(f'<div class="card" style="font-size:.82rem;line-height:1.85">{summary}</div>', unsafe_allow_html=True)

        with tab_exp:
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg_json = save_config(
                prompt_a=prompt_a, prompt_b="", task=task, model=model,
                temperature=temperature, n_runs=1, max_tokens=max_tokens,
                temperatures=temperatures, json_mode=json_mode, mode=mode,
            )
            st.download_button("💾 Save Config", cfg_json, f"config_{_ts}.json", "application/json", use_container_width=True)


    # ── PROMPT COMPARISON ──────────────────────────────────────────────────
    else:
        pb_a = st.progress(0)
        results_a = run_batch(client, prompt_a, task, model, temperature, n_runs, max_tokens, json_mode, pb_a, "Prompt A")
        pb_b = st.progress(0)
        results_b = run_batch(client, prompt_b, task, model, temperature, n_runs, max_tokens, json_mode, pb_b, "Prompt B")

        texts_a = [r["text"] for r in results_a if not r.get("error")]
        texts_b = [r["text"] for r in results_b if not r.get("error")]

        sim_a  = pairwise_similarity(texts_a) if texts_a else {"avg_sim": 0}
        sim_b  = pairwise_similarity(texts_b) if texts_b else {"avg_sim": 0}
        lst_a  = length_stats(texts_a) if texts_a else {}
        lst_b  = length_stats(texts_b) if texts_b else {}
        di_a   = determinism_index(sim_a["avg_sim"], lst_a.get("cv",0)) if texts_a else {"di":0,"label":"N/A","interpretation":""}
        di_b   = determinism_index(sim_b["avg_sim"], lst_b.get("cv",0)) if texts_b else {"di":0,"label":"N/A","interpretation":""}
        outs_a = detect_outliers(texts_a, outlier_thresh)
        outs_b = detect_outliers(texts_b, outlier_thresh)

        tab_raw, tab_met, tab_cmp, tab_ana, tab_exp = st.tabs([
            "📋 Raw Outputs", "📊 Metrics", "🏆 Comparison", "🔬 Analysis", "📤 Export"
        ])

        with tab_raw:
            ca, cb = st.columns(2, gap="large")
            with ca:
                st.markdown('<p class="sec" style="color:#38bdf8">Prompt A</p>', unsafe_allow_html=True)
                for i, r in enumerate(results_a):
                    extra = " outlier" if i in outs_a else ""
                    lbl = f"RUN {i+1}" + (" ⚠️" if i in outs_a else "")
                    if r.get("error"):
                        st.markdown(f'<div class="resp err"><div class="run-lbl">{lbl}</div>{r["error"]}</div>', unsafe_allow_html=True)
                    else:
                        meta = f"{len(r['text'].split())} words · {r['latency_ms']}ms"
                        st.markdown(f'<div class="resp a{extra}"><div class="run-lbl">{lbl} | {meta}</div>{r["text"]}</div>', unsafe_allow_html=True)
            with cb:
                st.markdown('<p class="sec" style="color:#fb7185">Prompt B</p>', unsafe_allow_html=True)
                for i, r in enumerate(results_b):
                    extra = " outlier" if i in outs_b else ""
                    lbl = f"RUN {i+1}" + (" ⚠️" if i in outs_b else "")
                    if r.get("error"):
                        st.markdown(f'<div class="resp err"><div class="run-lbl">{lbl}</div>{r["error"]}</div>', unsafe_allow_html=True)
                    else:
                        meta = f"{len(r['text'].split())} words · {r['latency_ms']}ms"
                        st.markdown(f'<div class="resp b{extra}"><div class="run-lbl">{lbl} | {meta}</div>{r["text"]}</div>', unsafe_allow_html=True)

        with tab_met:
            st.markdown('<p class="sec" style="color:#38bdf8">Prompt A</p>', unsafe_allow_html=True)
            metric_row([
                (f"{sim_a['avg_sim']}%", "Consistency",       "#38bdf8"),
                (str(di_a["di"]),         "Determinism Index", "#38bdf8"),
                (str(lst_a.get("mean",0)),"Avg Words",         "#38bdf8"),
                (str(lst_a.get("std_dev",0)),"Std Dev",        "#38bdf8"),
            ])
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="sec" style="color:#fb7185">Prompt B</p>', unsafe_allow_html=True)
            metric_row([
                (f"{sim_b['avg_sim']}%", "Consistency",       "#fb7185"),
                (str(di_b["di"]),         "Determinism Index", "#fb7185"),
                (str(lst_b.get("mean",0)),"Avg Words",         "#fb7185"),
                (str(lst_b.get("std_dev",0)),"Std Dev",        "#fb7185"),
            ])
            if texts_a and texts_b:
                import pandas as pd
                st.markdown("<br>", unsafe_allow_html=True)
                chart = {f"Run {i+1}": {"Prompt A": lst_a["lengths"][i], "Prompt B": lst_b["lengths"][i]}
                         for i in range(min(len(lst_a.get("lengths",[])), len(lst_b.get("lengths",[]))))}
                df = pd.DataFrame(chart).T
                st.bar_chart(df, color=["#38bdf8","#fb7185"], height=180)

        with tab_cmp:
            st.markdown('<p class="sec">🏆 Evaluation Report</p>', unsafe_allow_html=True)
            if texts_a and texts_b:
                # Auto verdicts
                verdicts = []
                if di_a["di"] > di_b["di"]:
                    verdicts.append(f"✅ **Prompt A** is more deterministic (DI: {di_a['di']} vs {di_b['di']})")
                else:
                    verdicts.append(f"✅ **Prompt B** is more deterministic (DI: {di_b['di']} vs {di_a['di']})")

                if lst_a.get("mean",0) > lst_b.get("mean",0):
                    verdicts.append(f"📝 **Prompt A** produces longer, more detailed responses ({lst_a['mean']} vs {lst_b['mean']} words avg)")
                else:
                    verdicts.append(f"📝 **Prompt B** produces longer responses ({lst_b['mean']} vs {lst_a['mean']} words avg)")

                if lst_a.get("std_dev",99) < lst_b.get("std_dev",99):
                    verdicts.append(f"📐 **Prompt A** has more predictable output length (σ={lst_a['std_dev']} vs {lst_b['std_dev']})")
                else:
                    verdicts.append(f"📐 **Prompt B** has more predictable output length (σ={lst_b['std_dev']} vs {lst_a['std_dev']})")

                # Hallucination
                if show_hallucination:
                    avg_ha = sum(hallucination_risk(t)["score"] for t in texts_a) / len(texts_a)
                    avg_hb = sum(hallucination_risk(t)["score"] for t in texts_b) / len(texts_b)
                    safer = "A" if avg_ha <= avg_hb else "B"
                    verdicts.append(f"⚠️ **Prompt {safer}** has lower hallucination risk (avg score: A={round(avg_ha)} B={round(avg_hb)})")

                # Adherence
                if show_adherence:
                    avg_aa = sum(instruction_adherence(prompt_a, t)["score"] for t in texts_a) / len(texts_a)
                    avg_ab = sum(instruction_adherence(prompt_b, t)["score"] for t in texts_b) / len(texts_b)
                    better = "A" if avg_aa >= avg_ab else "B"
                    verdicts.append(f"📋 **Prompt {better}** follows instructions better (adherence: A={round(avg_aa)} B={round(avg_ab)})")

                for v in verdicts:
                    st.markdown(v)

                # AI Comparison Report
                if show_ai_summary:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="sec">🧠 AI Comparison Report</p>', unsafe_allow_html=True)
                    with st.spinner("Generating analytical report…"):
                        summary = generate_summary(
                            client=client, model=model, mode=mode, task=task,
                            prompt_a=prompt_a, prompt_b=prompt_b,
                            texts_a=texts_a, texts_b=texts_b,
                            metrics_a=lst_a, metrics_b=lst_b,
                            di_a=di_a, di_b=di_b,
                            temperatures=temperatures,
                        )
                    st.markdown(f'<div class="card" style="font-size:.82rem;line-height:1.85">{summary}</div>', unsafe_allow_html=True)

        with tab_ana:
            if show_diff and texts_a and texts_b:
                st.markdown('<p class="sec">🔍 Word Diff — A[Run1] vs B[Run1]</p>', unsafe_allow_html=True)
                diff = word_diff(texts_a[0], texts_b[0])
                html = " ".join(
                    f'<span class="dadd">{c["text"]}</span>' if c["type"]=="added"
                    else f'<span class="drem">{c["text"]}</span>' if c["type"]=="removed"
                    else c["text"]
                    for c in diff
                )
                st.markdown(f'<div class="card" style="font-size:.8rem;line-height:1.9">{html}</div>', unsafe_allow_html=True)

        with tab_exp:
            _exp = build_experiment_dict(
                prompt_a=prompt_a, prompt_b=prompt_b, task=task, model=model,
                temperature=temperature, n_runs=n_runs,
                responses_a=results_a, responses_b=results_b,
                metrics_a=lst_a, metrics_b=lst_b,
                di_a=di_a, di_b=di_b,
                ai_summary=summary if show_ai_summary and texts_a else "",
                mode=mode,
            )
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.download_button("⬇️ JSON",     to_json(_exp),     f"pl_{_ts}.json", "application/json",  use_container_width=True)
            with c2:
                st.download_button("⬇️ CSV",      to_csv(_exp),      f"pl_{_ts}.csv",  "text/csv",          use_container_width=True)
            with c3:
                st.download_button("⬇️ Markdown", to_markdown(_exp), f"pl_{_ts}.md",   "text/markdown",     use_container_width=True)
            with c4:
                cfg_json = save_config(
                    prompt_a=prompt_a, prompt_b=prompt_b, task=task, model=model,
                    temperature=temperature, n_runs=n_runs, max_tokens=max_tokens,
                    temperatures=temperatures, json_mode=json_mode, mode=mode,
                )
                st.download_button("💾 Config",   cfg_json,          f"config_{_ts}.json","application/json",use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<br><br>
<p style="text-align:center;color:#1e2a3a;font-family:'JetBrains Mono',monospace;font-size:.6rem;letter-spacing:2px">
PROMPTLAB PRO v{APP_VERSION} · OPENROUTER · STREAMLIT
</p>
""", unsafe_allow_html=True)