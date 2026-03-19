"""
PromptLab Pro — app.py
========================
Senior-grade Prompt Evaluation & Analysis System
Powered by OpenRouter | Built with Streamlit
"""
from __future__ import annotations

import streamlit as st
import os, json, time
from datetime import datetime
from dotenv import load_dotenv

# ── Local modules ────────────────────────────────────────────────────────────
from modules.llm_client import get_client, call_llm
from modules.analysis import (
    consistency_score, hallucination_risk,
    validate_json, length_stats, word_diff, compare_prompts,
)
from modules.export import (
    to_json, to_csv, to_markdown, build_experiment_dict,
)
from modules.templates import TEMPLATES

# ────────────────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="PromptLab Pro",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

:root {
    --bg:       #07090f;
    --surface:  #0d1017;
    --card:     #111520;
    --border:   #1c2333;
    --a:        #4f9eff;
    --b:        #ff6b9d;
    --g:        #3effa0;
    --y:        #ffd166;
    --text:     #d0d8f0;
    --muted:    #4a5580;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg); }

/* Header */
.plp-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.plp-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--a);
    letter-spacing: -0.5px;
    margin: 0;
}
.plp-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Section headers */
.sec-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Cards */
.pl-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}

/* Response card */
.resp-card {
    background: var(--surface);
    border-left: 3px solid var(--a);
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    border-radius: 0 6px 6px 0;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    line-height: 1.75;
    white-space: pre-wrap;
    word-break: break-word;
}
.resp-card.b { border-left-color: var(--b); }
.resp-card.error { border-left-color: #ff4444; opacity: 0.7; }

/* Run label */
.run-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

/* Metric box */
.mbox {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 0.8rem;
    text-align: center;
}
.mbox-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    line-height: 1;
}
.mbox-lbl {
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: var(--muted);
    margin-top: 0.3rem;
    text-transform: uppercase;
}

/* Risk / consistency badges */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-weight: 600;
}
.badge-green  { background: rgba(62,255,160,0.12); color: var(--g); border: 1px solid rgba(62,255,160,0.3); }
.badge-yellow { background: rgba(255,209,102,0.12); color: var(--y); border: 1px solid rgba(255,209,102,0.3); }
.badge-red    { background: rgba(255,75,75,0.12); color: #ff6b6b; border: 1px solid rgba(255,75,75,0.3); }

/* Diff */
.diff-add { background: rgba(62,255,160,0.1); color: var(--g); padding: 0 3px; border-radius: 3px; }
.diff-rem { background: rgba(255,107,157,0.1); color: var(--b); padding: 0 3px; border-radius: 3px; text-decoration: line-through; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
}
.stTextArea textarea:focus {
    border-color: var(--a) !important;
    box-shadow: 0 0 0 2px rgba(79,158,255,0.15) !important;
}

/* Button */
.stButton > button {
    background: var(--a) !important;
    color: #07090f !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Selectbox */
.stSelectbox [data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

label { color: var(--muted) !important; font-size: 0.68rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; }

div[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--a) !important;
    border-bottom: 2px solid var(--a) !important;
}

/* Meta tags */
.meta-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--muted);
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="plp-header">
    <p class="plp-sub">⚗️ Prompt Evaluation System</p>
    <h1 class="plp-title">PromptLab Pro</h1>
    <p class="plp-sub">Consistency · Hallucination Detection · Temperature Sweep · Export</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar — Config ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sec-head">🎛️ Configuration</p>', unsafe_allow_html=True)

    model = st.selectbox("Model", options=[
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
    ], index=0)

    st.markdown("---")
    st.markdown('<p class="sec-head">⚙️ Parameters</p>', unsafe_allow_html=True)

    if mode == "🌡️ Temperature Sweep":
        temps_input = st.text_input(
            "Temperatures (comma-separated)",
            value="0.0, 0.3, 0.7, 1.0, 1.5",
            help="e.g. 0.0, 0.5, 1.0"
        )
        try:
            temperatures = [float(x.strip()) for x in temps_input.split(",")]
        except:
            temperatures = [0.0, 0.7, 1.0]
        n_runs = 1
        temperature = 0.7
    else:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        n_runs = st.select_slider("Runs per Prompt", options=[2, 3, 4, 5], value=3)
        temperatures = [temperature]

    max_tokens = st.slider("Max Tokens", 128, 1024, 512, 64)

    st.markdown("---")
    st.markdown('<p class="sec-head">🔧 Options</p>', unsafe_allow_html=True)

    json_mode = st.toggle("Force JSON Output", value=False)
    show_hallucination = st.toggle("Hallucination Heuristic", value=True)
    show_diff = st.toggle("Show Word Diff", value=True)

    st.markdown("---")
    st.markdown('<p class="sec-head">📚 Template Library</p>', unsafe_allow_html=True)

    template_choice = st.selectbox(
        "Load Template",
        options=["— Custom —"] + list(TEMPLATES.keys()),
        index=0,
    )


# ── Load Template ─────────────────────────────────────────────────────────────
selected_template = None
if template_choice != "— Custom —":
    selected_template = TEMPLATES[template_choice]


# ── Main Input Area ───────────────────────────────────────────────────────────
st.markdown('<p class="sec-head">📝 Inputs</p>', unsafe_allow_html=True)

task = st.text_area(
    "Task / Question",
    value=selected_template["example_task"] if selected_template else "",
    placeholder="e.g. What is recursion in programming?",
    height=70,
)

if mode == "🔁 Multi-Run Variability":
    prompt_a = st.text_area(
        "Prompt Template (use {task} as placeholder)",
        value=selected_template["template_a"] if selected_template else "",
        placeholder="You are a teacher. Explain {task} step by step.",
        height=120,
    )
    prompt_b = None
elif mode == "🌡️ Temperature Sweep":
    prompt_a = st.text_area(
        "Prompt Template (use {task} as placeholder)",
        value=selected_template["template_a"] if selected_template else "",
        placeholder="Explain {task} clearly.",
        height=120,
    )
    prompt_b = None
else:  # Prompt Comparison
    col_pa, col_pb = st.columns(2, gap="medium")
    with col_pa:
        prompt_a = st.text_area(
            "Prompt A",
            value=selected_template["template_a"] if selected_template else "",
            placeholder="You are a teacher. Explain {task} step by step.",
            height=120,
        )
    with col_pb:
        prompt_b = st.text_area(
            "Prompt B",
            value=selected_template["template_b"] if selected_template else "",
            placeholder="In one sentence, define {task}.",
            height=120,
        )

run_btn = st.button("⚗️ Run Experiment", use_container_width=True)


# ── Helper: build full prompt ─────────────────────────────────────────────────
def build_prompt(template: str, task_text: str) -> str:
    if "{task}" in template:
        return template.replace("{task}", task_text)
    return f"{template}\n\nTask: {task_text}"


# ── Helper: render response card ─────────────────────────────────────────────
def render_response(result: dict, run_idx: int, css_class: str = "a", temp: float | None = None):
    label = f"RUN {run_idx + 1}"
    if temp is not None:
        label = f"TEMP {temp}"

    if result.get("error"):
        st.markdown(f"""
        <div class="resp-card error">
            <div class="run-label">{label} — ERROR</div>
            {result['error']}
        </div>""", unsafe_allow_html=True)
        return

    meta = f"{len(result['text'].split())} words · {result['latency_ms']}ms"
    if result.get("tokens_completion"):
        meta += f" · {result['tokens_completion']} tokens"

    st.markdown(f"""
    <div class="resp-card {css_class}">
        <div class="run-label">{label} &nbsp;|&nbsp; {meta}</div>
        {result['text']}
    </div>""", unsafe_allow_html=True)


# ── Helper: render metrics row ────────────────────────────────────────────────
def render_metrics_row(stats: dict, cons: dict, color: str):
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, stats["mean"],    "Avg Words"),
        (c2, stats["std_dev"], "Std Dev"),
        (c3, stats["range"],   "Range"),
        (c4, f"{cons['score']}%", "Consistency"),
        (c5, stats["stability"].split()[0], "Stability"),
    ]:
        with col:
            st.markdown(f"""
            <div class="mbox">
                <div class="mbox-val" style="color:{color}">{val}</div>
                <div class="mbox-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)


# ── RUN ───────────────────────────────────────────────────────────────────────
if run_btn:
    if not task.strip():
        st.warning("⚠️ Enter a task/question first.")
        st.stop()
    if not prompt_a or not prompt_a.strip():
        st.warning("⚠️ Enter a prompt template.")
        st.stop()
    if mode == "🆚 Prompt Comparison" and (not prompt_b or not prompt_b.strip()):
        st.warning("⚠️ Enter Prompt B for comparison mode.")
        st.stop()

    client = get_client()

    # Experiment metadata
    exp_meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "temperature": temperature,
        "n_runs": n_runs,
        "mode": mode,
        "task": task,
    }

    st.markdown("---")
    # Metadata tags
    st.markdown(
        f'<span class="meta-tag">🤖 {model}</span>'
        f'<span class="meta-tag">🌡️ temp={temperature}</span>'
        f'<span class="meta-tag">🔁 runs={n_runs}</span>'
        f'<span class="meta-tag">📅 {exp_meta["timestamp"]}</span>'
        + ('<span class="meta-tag">📋 JSON mode</span>' if json_mode else ""),
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── MODE: Multi-Run Variability ──────────────────────────────────────
    if mode == "🔁 Multi-Run Variability":
        st.markdown('<p class="sec-head">🔬 Running Experiment</p>', unsafe_allow_html=True)
        full_prompt = build_prompt(prompt_a, task)
        pb = st.progress(0, text="Running…")
        results = []
        for i in range(n_runs):
            pb.progress((i + 0.5) / n_runs, text=f"Run {i+1}/{n_runs}…")
            r = call_llm(client, full_prompt, model, temperature, max_tokens, json_mode)
            results.append(r)
            time.sleep(0.1)
        pb.progress(1.0, text="Done ✓")

        texts = [r["text"] for r in results if not r["error"]]

        # ── Tabs: Raw / Analysis / Diff ──────────────────────────────
        tab1, tab2, tab3 = st.tabs(["📋 Raw Outputs", "📊 Analysis", "🔍 Diff Viewer"])

        with tab1:
            st.markdown('<p class="sec-head">Raw Outputs</p>', unsafe_allow_html=True)
            for i, r in enumerate(results):
                render_response(r, i, "a")

                if json_mode and not r["error"]:
                    jv = validate_json(r["text"])
                    badge_cls = "badge-green" if jv["valid"] else "badge-red"
                    badge_txt = "✓ Valid JSON" if jv["valid"] else f"✗ Invalid JSON — {jv['error']}"
                    st.markdown(f'<span class="badge {badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<p class="sec-head">Metrics</p>', unsafe_allow_html=True)
            if texts:
                stats  = length_stats(texts)
                cons   = consistency_score(texts)
                render_metrics_row(stats, cons, "#4f9eff")

                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="mbox">
                        <div class="mbox-val" style="color:#4f9eff;font-size:1rem">{cons['label']}</div>
                        <div class="mbox-lbl">Consistency Rating</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="mbox">
                        <div class="mbox-val" style="color:#4f9eff;font-size:1rem">{stats['stability']}</div>
                        <div class="mbox-lbl">Stability Rating</div>
                    </div>""", unsafe_allow_html=True)

                # Hallucination per run
                if show_hallucination:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="sec-head">⚠️ Hallucination Risk (per run)</p>', unsafe_allow_html=True)
                    for i, t in enumerate(texts):
                        hr = hallucination_risk(t)
                        cls = "badge-green" if "Low" in hr["level"] else ("badge-yellow" if "Medium" in hr["level"] else "badge-red")
                        st.markdown(f'**Run {i+1}:** <span class="badge {cls}">{hr["level"]}</span> (score: {hr["score"]})', unsafe_allow_html=True)
                        if hr["flags"]:
                            with st.expander(f"Run {i+1} flags"):
                                for f in hr["flags"]:
                                    st.markdown(f"- {f}")

                # Length distribution chart
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="sec-head">📈 Length Distribution</p>', unsafe_allow_html=True)
                import pandas as pd
                df = pd.DataFrame({"Words": stats["lengths"]}, index=[f"Run {i+1}" for i in range(len(stats["lengths"]))])
                st.bar_chart(df, color="#4f9eff", height=180)

        with tab3:
            if show_diff and len(texts) >= 2:
                st.markdown('<p class="sec-head">Word-level Diff: Run 1 vs Run 2</p>', unsafe_allow_html=True)
                diff = word_diff(texts[0], texts[1])
                html_parts = []
                for chunk in diff:
                    if chunk["type"] == "equal":
                        html_parts.append(chunk["text"])
                    elif chunk["type"] == "added":
                        html_parts.append(f'<span class="diff-add">{chunk["text"]}</span>')
                    elif chunk["type"] == "removed":
                        html_parts.append(f'<span class="diff-rem">{chunk["text"]}</span>')
                st.markdown(
                    f'<div class="pl-card" style="font-size:0.82rem;line-height:1.8">{" ".join(html_parts)}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Need at least 2 successful runs for diff view.")

        # Store for export
        st.session_state["last_experiment"] = {
            "mode": mode,
            "metadata": exp_meta,
            "prompt_a": prompt_a,
            "prompt_b": "",
            "results_a": results,
            "results_b": [],
            "texts_a": texts,
            "texts_b": [],
        }

    # ── MODE: Temperature Sweep ──────────────────────────────────────────
    elif mode == "🌡️ Temperature Sweep":
        st.markdown('<p class="sec-head">🌡️ Temperature Sweep Results</p>', unsafe_allow_html=True)
        full_prompt = build_prompt(prompt_a, task)

        sweep_results = {}
        pb = st.progress(0, text="Starting sweep…")
        for ti, temp in enumerate(temperatures):
            pb.progress((ti + 0.5) / len(temperatures), text=f"Temperature {temp}…")
            r = call_llm(client, full_prompt, model, temp, max_tokens, json_mode)
            sweep_results[temp] = r
            time.sleep(0.1)
        pb.progress(1.0, text="Sweep complete ✓")

        st.markdown("<br>", unsafe_allow_html=True)

        for temp, r in sweep_results.items():
            col_temp, col_resp = st.columns([1, 5], gap="medium")
            with col_temp:
                color = "#4f9eff" if temp <= 0.5 else ("#ffd166" if temp <= 1.0 else "#ff6b9d")
                st.markdown(f"""
                <div class="mbox" style="margin-top:0.5rem">
                    <div class="mbox-val" style="color:{color}">{temp}</div>
                    <div class="mbox-lbl">temp</div>
                    <div style="color:{color};font-size:0.65rem;margin-top:0.3rem">
                        {"deterministic" if temp == 0 else ("balanced" if temp <= 0.7 else "creative")}
                    </div>
                </div>""", unsafe_allow_html=True)
            with col_resp:
                render_response(r, 0, "a", temp=temp)

        # Sweep analysis
        valid_texts = {t: r["text"] for t, r in sweep_results.items() if not r.get("error")}
        if len(valid_texts) >= 2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="sec-head">📊 Sweep Analysis</p>', unsafe_allow_html=True)
            import pandas as pd
            lengths_by_temp = {str(t): len(txt.split()) for t, txt in valid_texts.items()}
            df = pd.DataFrame({"Words": list(lengths_by_temp.values())}, index=list(lengths_by_temp.keys()))
            st.bar_chart(df, color="#4f9eff", height=180)

    # ── MODE: Prompt Comparison ──────────────────────────────────────────
    else:
        st.markdown('<p class="sec-head">🔬 Running Comparison</p>', unsafe_allow_html=True)
        full_a = build_prompt(prompt_a, task)
        full_b = build_prompt(prompt_b, task)

        results_a, results_b = [], []
        pb_a = st.progress(0, text="Prompt A…")
        for i in range(n_runs):
            pb_a.progress((i + 0.5) / n_runs, text=f"Prompt A — run {i+1}/{n_runs}")
            results_a.append(call_llm(client, full_a, model, temperature, max_tokens, json_mode))
            time.sleep(0.1)
        pb_a.progress(1.0, text="Prompt A done ✓")

        pb_b = st.progress(0, text="Prompt B…")
        for i in range(n_runs):
            pb_b.progress((i + 0.5) / n_runs, text=f"Prompt B — run {i+1}/{n_runs}")
            results_b.append(call_llm(client, full_b, model, temperature, max_tokens, json_mode))
            time.sleep(0.1)
        pb_b.progress(1.0, text="Prompt B done ✓")

        texts_a = [r["text"] for r in results_a if not r.get("error")]
        texts_b = [r["text"] for r in results_b if not r.get("error")]

        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Raw Outputs", "📊 Metrics", "🏆 Comparison Summary", "🔍 Diff"
        ])

        with tab1:
            col_a, col_b = st.columns(2, gap="large")
            with col_a:
                st.markdown('<p class="sec-head" style="color:#4f9eff">Prompt A</p>', unsafe_allow_html=True)
                for i, r in enumerate(results_a):
                    render_response(r, i, "a")
            with col_b:
                st.markdown('<p class="sec-head" style="color:#ff6b9d">Prompt B</p>', unsafe_allow_html=True)
                for i, r in enumerate(results_b):
                    render_response(r, i, "b")

        with tab2:
            if texts_a and texts_b:
                stats_a = length_stats(texts_a)
                stats_b = length_stats(texts_b)
                cons_a  = consistency_score(texts_a)
                cons_b  = consistency_score(texts_b)

                st.markdown('<p class="sec-head" style="color:#4f9eff">Prompt A Metrics</p>', unsafe_allow_html=True)
                render_metrics_row(stats_a, cons_a, "#4f9eff")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="sec-head" style="color:#ff6b9d">Prompt B Metrics</p>', unsafe_allow_html=True)
                render_metrics_row(stats_b, cons_b, "#ff6b9d")

                # Hallucination
                if show_hallucination:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="sec-head">⚠️ Hallucination Risk</p>', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    for col, texts, label, color in [
                        (c1, texts_a, "Prompt A", "#4f9eff"),
                        (c2, texts_b, "Prompt B", "#ff6b9d"),
                    ]:
                        with col:
                            avg_risk = sum(hallucination_risk(t)["score"] for t in texts) / len(texts)
                            level = "Low Risk 🟢" if avg_risk < 25 else ("Medium Risk 🟡" if avg_risk < 55 else "High Risk 🔴")
                            cls = "badge-green" if avg_risk < 25 else ("badge-yellow" if avg_risk < 55 else "badge-red")
                            st.markdown(f"""
                            <div class="mbox">
                                <div class="mbox-val" style="color:{color};font-size:1rem">{level}</div>
                                <div class="mbox-lbl">{label} Avg Risk Score: {round(avg_risk)}</div>
                            </div>""", unsafe_allow_html=True)

                # Length chart
                st.markdown("<br>", unsafe_allow_html=True)
                import pandas as pd
                chart = {}
                for i in range(min(len(stats_a["lengths"]), len(stats_b["lengths"]))):
                    chart[f"Run {i+1}"] = {"Prompt A": stats_a["lengths"][i], "Prompt B": stats_b["lengths"][i]}
                df = pd.DataFrame(chart).T
                st.bar_chart(df, color=["#4f9eff", "#ff6b9d"], height=200)

        with tab3:
            st.markdown('<p class="sec-head">🏆 Which Prompt Performed Better?</p>', unsafe_allow_html=True)
            if texts_a and texts_b:
                summary = compare_prompts(texts_a, texts_b, "Prompt A", "Prompt B")
                st.markdown(f'<div class="pl-card">{summary}</div>', unsafe_allow_html=True)

                # Detailed verdict
                cons_a = consistency_score(texts_a)
                cons_b = consistency_score(texts_b)
                stats_a = length_stats(texts_a)
                stats_b = length_stats(texts_b)

                st.markdown("<br>", unsafe_allow_html=True)
                verdicts = []
                if cons_a["score"] > cons_b["score"]:
                    verdicts.append("✅ **Prompt A** is more consistent across runs")
                else:
                    verdicts.append("✅ **Prompt B** is more consistent across runs")

                if stats_a["mean"] > stats_b["mean"]:
                    verdicts.append("📝 **Prompt A** produces more detailed responses")
                else:
                    verdicts.append("📝 **Prompt B** produces more detailed responses")

                if stats_a["std_dev"] < stats_b["std_dev"]:
                    verdicts.append("📐 **Prompt A** has more predictable length")
                else:
                    verdicts.append("📐 **Prompt B** has more predictable length")

                for v in verdicts:
                    st.markdown(v)

        with tab4:
            if show_diff and texts_a and texts_b:
                st.markdown('<p class="sec-head">Word-level Diff: Prompt A Run 1 vs Prompt B Run 1</p>', unsafe_allow_html=True)
                diff = word_diff(texts_a[0], texts_b[0])
                html_parts = []
                for chunk in diff:
                    if chunk["type"] == "equal":
                        html_parts.append(chunk["text"])
                    elif chunk["type"] == "added":
                        html_parts.append(f'<span class="diff-add">{chunk["text"]}</span>')
                    elif chunk["type"] == "removed":
                        html_parts.append(f'<span class="diff-rem">{chunk["text"]}</span>')
                st.markdown(
                    f'<div class="pl-card" style="font-size:0.82rem;line-height:1.8">{" ".join(html_parts)}</div>',
                    unsafe_allow_html=True
                )

        # Store for export
        st.session_state["last_experiment"] = {
            "mode": mode,
            "metadata": exp_meta,
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "results_a": results_a,
            "results_b": results_b,
            "texts_a": texts_a,
            "texts_b": texts_b,
        }

    # ── Export Section ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="sec-head">📤 Export Results</p>', unsafe_allow_html=True)

    exp = st.session_state.get("last_experiment", {})
    if exp:
        texts_a = exp.get("texts_a", [])
        texts_b = exp.get("texts_b", [])

        stats_a  = length_stats(texts_a) if texts_a else {}
        stats_b  = length_stats(texts_b) if texts_b else {}
        cons_a   = consistency_score(texts_a) if texts_a else {"score": 0, "label": "N/A"}
        cons_b   = consistency_score(texts_b) if texts_b else {"score": 0, "label": "N/A"}

        exp_dict = build_experiment_dict(
            prompt_a=exp.get("prompt_a", ""),
            prompt_b=exp.get("prompt_b", ""),
            task=task,
            model=model,
            temperature=temperature,
            n_runs=n_runs,
            responses_a=exp.get("results_a", []),
            responses_b=exp.get("results_b", []),
            metrics_a=stats_a,
            metrics_b=stats_b,
            consistency_a=cons_a,
            consistency_b=cons_b,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ Export JSON",
                data=to_json(exp_dict),
                file_name=f"promptlab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "⬇️ Export CSV",
                data=to_csv(exp_dict),
                file_name=f"promptlab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c3:
            st.download_button(
                "⬇️ Export Markdown",
                data=to_markdown(exp_dict),
                file_name=f"promptlab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<br><br>
<p style="text-align:center;color:#1c2333;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;letter-spacing:2px">
PROMPTLAB PRO · OPENROUTER · STREAMLIT
</p>
""", unsafe_allow_html=True)