"""
Prompt Evaluation & Variability Testing Lab
============================================
An LLM experimentation tool to study how prompt structure and temperature
affect the stability and variability of LLM responses.

>> Powered by DeepSeek API (OpenAI-compatible) <<
"""

from __future__ import annotations

import streamlit as st
import os
import time
import math
from openai import OpenAI
from dotenv import load_dotenv

# ── Optional: similarity scoring via sentence-transformers ──────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Lab",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --border:    #1e1e2e;
    --accent-a:  #7c6af7;
    --accent-b:  #f76a8a;
    --accent-g:  #4af7b0;
    --text:      #e8e8f0;
    --muted:     #6b6b80;
    --card:      #13131c;
}

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg); }

.lab-header {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.lab-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent-a) 0%, var(--accent-b) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.lab-sub {
    color: var(--muted);
    font-size: 0.8rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

.badge-a { color: var(--accent-a); font-weight: 700; }
.badge-b { color: var(--accent-b); font-weight: 700; }
.badge-g { color: var(--accent-g); font-weight: 700; }

.response-card {
    background: var(--surface);
    border-left: 3px solid var(--accent-a);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 0.6rem 0;
    font-size: 0.82rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    animation: fadeIn 0.4s ease;
}
.response-card.b { border-left-color: var(--accent-b); }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

.run-num {
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 0.35rem;
}

.metric-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: var(--muted);
    margin-top: 0.3rem;
    text-transform: uppercase;
}

.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-a) !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.2) !important;
}
label { color: var(--muted) !important; font-size: 0.72rem !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }

.stSlider [data-baseweb="slider"] { padding-top: 0.5rem; }
.stSlider [data-baseweb="thumb"] { background: var(--accent-a) !important; }
.stSlider [data-baseweb="track-fill"] { background: var(--accent-a) !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent-a), var(--accent-b)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    padding: 0.7rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSelectbox [data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

div[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

.stAlert { border-radius: 8px !important; }

.diff-added   { background: rgba(74,247,176,0.12); border-left: 3px solid var(--accent-g); padding: 0.3rem 0.6rem; margin: 0.3rem 0; border-radius: 0 4px 4px 0; font-size: 0.78rem; }
.diff-removed { background: rgba(247,106,138,0.1);  border-left: 3px solid var(--accent-b); padding: 0.3rem 0.6rem; margin: 0.3rem 0; border-radius: 0 4px 4px 0; font-size: 0.78rem; }

/* DeepSeek badge */
.ds-badge {
    display: inline-block;
    background: linear-gradient(135deg, #4af7b0, #7c6af7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 2px;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lab-header">
    <p class="lab-sub">⚗️ LLM Experimentation Tool</p>
    <h1 class="lab-title">Prompt Evaluation Lab</h1>
    <p class="lab-sub">Variability · Stability · Temperature Analysis</p>
    <p class="lab-sub" style="margin-top:0.5rem">
        <span class="ds-badge">⚡ Powered by DeepSeek</span>
    </p>
</div>
""", unsafe_allow_html=True)


# ── API Client ───────────────────────────────────────────────────────────────
def configure_api() -> OpenAI:
    """
    Initialize DeepSeek client using OpenAI SDK.
    OpenRouter is fully OpenAI-compatible — free DeepSeek models available.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")
    if not api_key:
        st.error(
            "🔑 **OPENROUTER_API_KEY** not found.\n\n"
            "**Setup karo:**\n"
            "1. Jao → https://platform.deepseek.com\n"
            "2. Sign up karo (free — 5M tokens milenge)\n"
            "3. API Keys section mein nayi key banao\n"
            "4. `.env` file mein likho: `OPENROUTER_API_KEY=your_key_here`"
        )
        st.stop()

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:8501",  # Required by OpenRouter
            "X-Title": "Prompt Evaluation Lab",
        },
    )
    return client


# ── Core: single LLM call ────────────────────────────────────────────────────
def call_llm(client: OpenAI, prompt: str, temperature: float, model: str = "openrouter/free") -> str:
    """
    Send a single prompt to DeepSeek and return the response text.
    Uses the standard OpenAI chat completions format.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"


# ── Core: run N experiments for a prompt ─────────────────────────────────────
def run_experiment(
    client: OpenAI,
    prompt_template: str,
    task: str,
    temperature: float,
    n_runs: int,
    model: str,
    progress_label: str,
    progress_bar,
) -> list[str]:
    """
    Substitute the task into the prompt template and call the LLM n_runs times.
    Returns a list of response strings.
    """
    if "{task}" in prompt_template:
        full_prompt = prompt_template.replace("{task}", task)
    else:
        full_prompt = f"{prompt_template}\n\nTask: {task}"

    responses = []
    for i in range(n_runs):
        progress_bar.progress((i + 0.5) / n_runs, text=f"{progress_label} — run {i+1}/{n_runs}")
        result = call_llm(client, full_prompt, temperature, model)
        responses.append(result)
        time.sleep(0.5)  # small delay to avoid rate-limit spikes

    progress_bar.progress(1.0, text=f"{progress_label} — done ✓")
    return responses


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(responses: list[str]) -> dict:
    """Compute length-based statistics across a list of responses."""
    lengths = [len(r.split()) for r in responses]
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((x - mean) ** 2 for x in lengths) / n
    std_dev = math.sqrt(variance)
    return {
        "lengths":  lengths,
        "mean":     round(mean, 1),
        "variance": round(variance, 1),
        "std_dev":  round(std_dev, 2),
        "min":      min(lengths),
        "max":      max(lengths),
        "range":    max(lengths) - min(lengths),
    }


def compute_similarity(responses: list[str]) -> float | None:
    """
    Compute average pairwise cosine similarity using sentence-transformers.
    Returns None if the library is not installed.
    """
    if not SIMILARITY_AVAILABLE or len(responses) < 2:
        return None
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(responses)
        n = len(embeddings)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = embeddings[i], embeddings[j]
                cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                sims.append(cos)
        return round(sum(sims) / len(sims), 4) if sims else None
    except Exception:
        return None


def stability_rating(std_dev: float, mean: float) -> tuple[str, str]:
    """Return a human-readable stability label and colour based on CV."""
    if mean == 0:
        return "N/A", "#6b6b80"
    cv = std_dev / mean
    if cv < 0.05:
        return "Very Stable 🟢", "#4af7b0"
    if cv < 0.15:
        return "Stable 🟡", "#f7d96a"
    if cv < 0.30:
        return "Variable 🟠", "#f7a96a"
    return "Highly Variable 🔴", "#f76a8a"


# ── UI: Metrics display ───────────────────────────────────────────────────────
def show_metrics(responses_a: list[str], responses_b: list[str]):
    """Render the analytics summary section comparing both prompts."""
    m_a = compute_metrics(responses_a)
    m_b = compute_metrics(responses_b)
    stab_a, col_a = stability_rating(m_a["std_dev"], m_a["mean"])
    stab_b, col_b = stability_rating(m_b["std_dev"], m_b["mean"])

    st.markdown("---")
    st.markdown('<p class="section-label">📊 Analytics Summary</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for col, m, label, badge_class, color in [
        (col1, m_a, "Prompt A", "badge-a", "#7c6af7"),
        (col2, m_b, "Prompt B", "badge-b", "#f76a8a"),
    ]:
        with col:
            st.markdown(
                f'<p class="section-label"><span class="{badge_class}">{label}</span> — Word Count Stats</p>',
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            for c, val, lbl in [
                (c1, m["mean"],    "Avg Words"),
                (c2, m["std_dev"], "Std Dev"),
                (c3, m["range"],   "Range"),
                (c4, m["variance"],"Variance"),
            ]:
                with c:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value" style="color:{color}">{val}</div>
                        <div class="metric-label">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:{col_a}; font-size:1.3rem">{stab_a}</div>
            <div class="metric-label">Prompt A — Stability Rating</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:{col_b}; font-size:1.3rem">{stab_b}</div>
            <div class="metric-label">Prompt B — Stability Rating</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sim_a = compute_similarity(responses_a)
    sim_b = compute_similarity(responses_b)

    cs1, cs2, cs3 = st.columns(3)
    with cs1:
        val = f"{sim_a:.3f}" if sim_a is not None else "N/A*"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:#7c6af7">{val}</div>
            <div class="metric-label">Prompt A — Avg Similarity</div>
        </div>""", unsafe_allow_html=True)
    with cs2:
        val = f"{sim_b:.3f}" if sim_b is not None else "N/A*"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:#f76a8a">{val}</div>
            <div class="metric-label">Prompt B — Avg Similarity</div>
        </div>""", unsafe_allow_html=True)
    with cs3:
        if sim_a is not None and sim_b is not None:
            winner = "A" if sim_a > sim_b else ("B" if sim_b > sim_a else "Tie")
            color  = "#7c6af7" if winner == "A" else ("#f76a8a" if winner == "B" else "#4af7b0")
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color:{color}">Prompt {winner}</div>
                <div class="metric-label">More Consistent</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value" style="color:#6b6b80; font-size:0.8rem">Install sentence-transformers</div>
                <div class="metric-label">for similarity scores</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">📈 Response Length Distribution</p>', unsafe_allow_html=True)

    import pandas as pd
    chart_data = {}
    for i, la in enumerate(m_a["lengths"]):
        chart_data[f"Run {i+1}"] = {"Prompt A": la, "Prompt B": m_b["lengths"][i]}
    df = pd.DataFrame(chart_data).T
    st.bar_chart(df, color=["#7c6af7", "#f76a8a"], height=220)


# ── UI: Responses display ─────────────────────────────────────────────────────
def show_responses(responses: list[str], label: str, css_class: str):
    """Render all responses for one prompt in styled cards."""
    badge = "badge-a" if css_class == "a" else "badge-b"
    st.markdown(
        f'<p class="section-label"><span class="{badge}">{label}</span> — {len(responses)} Runs</p>',
        unsafe_allow_html=True,
    )
    for i, resp in enumerate(responses):
        st.markdown(f"""
        <div class="response-card {css_class}">
            <div class="run-num">RUN {i+1}</div>
            {resp}
        </div>
        """, unsafe_allow_html=True)


# ── Main UI Layout ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">⚙️ Experiment Configuration</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 1], gap="large")

with col_left:
    task = st.text_area(
        "TASK / QUESTION",
        placeholder="e.g. Explain the concept of recursion in programming",
        height=80,
        help="The core question or task both prompts will try to answer.",
    )

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        prompt_a = st.text_area(
            "PROMPT TEMPLATE A",
            placeholder="You are a teacher. Explain {task} step by step.",
            height=160,
            help="Use {task} as a placeholder — it will be replaced with your task above.",
        )
    with col_b:
        prompt_b = st.text_area(
            "PROMPT TEMPLATE B",
            placeholder="In one paragraph, describe {task} simply.",
            height=160,
            help="Use {task} as a placeholder — it will be replaced with your task above.",
        )

with col_right:
    st.markdown('<p class="section-label">🎛️ Parameters</p>', unsafe_allow_html=True)

    temperature = st.slider(
        "TEMPERATURE",
        min_value=0.0, max_value=2.0, value=0.7, step=0.05,
        help="Higher = more random/creative. Lower = more deterministic.",
    )
    n_runs = st.select_slider(
        "NUMBER OF RUNS",
        options=[2, 3, 4, 5],
        value=3,
        help="How many times to run each prompt.",
    )

    # ── Free models on OpenRouter ────────────────────────────────────────
    model = st.selectbox(
        "MODEL",
        options=[
            "openrouter/free",                                    # Auto best free ✅
            "qwen/qwen-2.5-next-80b-a3b-instruct:free",          # Qwen 2.5 80B ✅
            "nvidia/nemotron-3-super-120b-a12b:free",             # Nvidia 120B ✅
            "openai/gpt-oss-120b:free",                           # GPT OSS 120B ✅
            "minimax/minimax-m2.5:free",                          # MiniMax ✅
        ],
        index=0,
        help=(
            "deepseek/deepseek-r1:free → Best for reasoning tasks\n"
            "meta-llama/llama-3.3-70b-instruct:free → Fast general purpose\n"
            "google/gemini-2.0-flash-exp:free → Google Gemini free\n"
            "openrouter/auto → Automatically picks best free model"
        ),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚗️ Run Experiment", use_container_width=True)


# ─ Run & Display ─────────────────────────────────────────────────────────────
if run_btn:
    if not task.strip():
        st.warning("⚠️ Please enter a task/question.")
        st.stop()
    if not prompt_a.strip() or not prompt_b.strip():
        st.warning("⚠️ Please fill in both Prompt Template A and B.")
        st.stop()

    client = configure_api()

    st.markdown("---")
    st.markdown('<p class="section-label">🔬 Running Experiment…</p>', unsafe_allow_html=True)

    pb_a = st.progress(0, text="Prompt A — starting…")
    responses_a = run_experiment(client, prompt_a, task, temperature, n_runs, model, "Prompt A", pb_a)

    pb_b = st.progress(0, text="Prompt B — starting…")
    responses_b = run_experiment(client, prompt_b, task, temperature, n_runs, model, "Prompt B", pb_b)

    # ── Results ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">📋 Results</p>', unsafe_allow_html=True)

    res_col_a, res_col_b = st.columns(2, gap="large")
    with res_col_a:
        show_responses(responses_a, "Prompt A", "a")
    with res_col_b:
        show_responses(responses_b, "Prompt B", "b")

    # ── Analytics ────────────────────────────────────────────────────────
    show_metrics(responses_a, responses_b)

    # ── Expandable: raw diff ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 First Response Diff (A vs B)"):
        if responses_a and responses_b:
            import difflib
            diff = list(difflib.unified_diff(
                responses_a[0].split(),
                responses_b[0].split(),
                lineterm="", n=0,
            ))
            if diff:
                for line in diff[2:]:
                    if line.startswith("+"):
                        st.markdown(f'<div class="diff-added">+ {line[1:]}</div>', unsafe_allow_html=True)
                    elif line.startswith("-"):
                        st.markdown(f'<div class="diff-removed">- {line[1:]}</div>', unsafe_allow_html=True)
            else:
                st.info("Responses are identical (word-level).")


# ─ Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<br><br>
<p style="text-align:center;color:#2a2a3a;font-size:0.7rem;letter-spacing:2px">
PROMPT EVALUATION LAB · POWERED BY OPENROUTER + DEEPSEEK · BUILT WITH STREAMLIT
</p>
""", unsafe_allow_html=True)