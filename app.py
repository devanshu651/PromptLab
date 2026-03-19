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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #080b12;
    --surf:    #0d1119;
    --card:    #111827;
    --border:  #1e2a3a;
    --a:       #38bdf8;
    --b:       #fb7185;
    --g:       #34d399;
    --y:       #fbbf24;
    --text:    #cbd5e1;
    --muted:   #475569;
    --code:    #94a3b8;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg); }

/* ── Layout ── */
.plp-header { padding: 1.5rem 0 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
.plp-title  { font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:var(--a); margin:0; }
.plp-ver    { font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:var(--muted); letter-spacing:2px; }
.sec        { font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:3px; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--border); padding-bottom:0.4rem; margin-bottom:1rem; }

/* ── Cards ── */
.card       { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1.2rem; margin-bottom:0.8rem; }
.resp       { background:var(--surf); border-left:3px solid var(--a); padding:0.9rem 1.1rem; margin:0.4rem 0; border-radius:0 6px 6px 0; font-size:0.8rem; line-height:1.8; white-space:pre-wrap; word-break:break-word; }
.resp.b     { border-left-color:var(--b); }
.resp.err   { border-left-color:#ef4444; opacity:.7; }
.run-lbl    { font-family:'JetBrains Mono',monospace; font-size:0.58rem; letter-spacing:2px; color:var(--muted); margin-bottom:0.35rem; }
.outlier    { border-left-color:var(--y) !important; }

/* ── Metrics ── */
.mbox       { background:var(--card); border:1px solid var(--border); border-radius:6px; padding:1rem 0.7rem; text-align:center; }
.mval       { font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; line-height:1; }
.mlbl       { font-size:0.58rem; letter-spacing:2px; color:var(--muted); margin-top:0.3rem; text-transform:uppercase; }

/* ── Badges ── */
.badge      { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:0.65rem; padding:0.2rem 0.5rem; border-radius:4px; font-weight:600; margin-right:0.3rem; }
.bg         { background:rgba(52,211,153,.12); color:var(--g); border:1px solid rgba(52,211,153,.3); }
.by         { background:rgba(251,191,36,.12);  color:var(--y); border:1px solid rgba(251,191,36,.3); }
.br         { background:rgba(251,113,133,.12); color:var(--b); border:1px solid rgba(251,113,133,.3); }
.ba         { background:rgba(56,189,248,.12);  color:var(--a); border:1px solid rgba(56,189,248,.3); }

/* ── Diff ── */
.dadd       { background:rgba(52,211,153,.15); color:var(--g); padding:0 3px; border-radius:2px; }
.drem       { background:rgba(251,113,133,.15); color:var(--b); padding:0 3px; border-radius:2px; text-decoration:line-through; }

/* ── Meta tags ── */
.mt         { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:0.6rem; padding:0.15rem 0.45rem; border-radius:3px; background:var(--card); border:1px solid var(--border); color:var(--muted); margin-right:0.3rem; margin-bottom:0.3rem; }

/* ── Why this matters box ── */
.why-box    { background:rgba(56,189,248,.06); border:1px solid rgba(56,189,248,.2); border-radius:8px; padding:1.2rem; font-size:0.82rem; line-height:1.75; }

/* ── Streamlit overrides ── */
section[data-testid="stSidebar"] { background:var(--surf) !important; border-right:1px solid var(--border) !important; }
.stTextArea textarea, .stTextInput input { background:var(--surf) !important; border:1px solid var(--border) !important; color:var(--text) !important; font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important; border-radius:6px !important; }
.stTextArea textarea:focus { border-color:var(--a) !important; box-shadow:0 0 0 2px rgba(56,189,248,.15) !important; }
.stButton > button { background:var(--a) !important; color:#080b12 !important; border:none !important; border-radius:6px !important; font-family:'JetBrains Mono',monospace !important; font-weight:700 !important; font-size:0.82rem !important; padding:0.55rem 1.2rem !important; width:100% !important; }
.stButton > button:hover { opacity:.85 !important; }
.stSelectbox [data-baseweb="select"] > div { background:var(--surf) !important; border-color:var(--border) !important; color:var(--text) !important; font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important; }
label { color:var(--muted) !important; font-size:0.64rem !important; letter-spacing:2px !important; text-transform:uppercase !important; }
div[data-testid="stExpander"] { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:8px !important; }
.stTabs [data-baseweb="tab-list"] { background:var(--surf) !important; border-bottom:1px solid var(--border) !important; gap:0 !important; }
.stTabs [data-baseweb="tab"] { font-family:'JetBrains Mono',monospace !important; font-size:0.68rem !important; letter-spacing:1.5px !important; color:var(--muted) !important; background:transparent !important; border:none !important; padding:0.55rem 1rem !important; }
.stTabs [aria-selected="true"] { color:var(--a) !important; border-bottom:2px solid var(--a) !important; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="plp-header">
    <p class="plp-ver">⚗️ PROMPT EVALUATION PLATFORM</p>
    <h1 class="plp-title">PromptLab Pro</h1>
    <p class="plp-ver">v{APP_VERSION} · Consistency · Determinism · Hallucination · Reproducibility</p>
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
                        color = "#34d399" if ia["score"] >= 75 else ("#fbbf24" if ia["score"] >= 50 else "#fb7185")
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
                color = "#38bdf8" if temp <= 0.3 else ("#fbbf24" if temp <= 1.0 else "#fb7185")
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