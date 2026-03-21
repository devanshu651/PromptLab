<div align="center">

<br/>

```
██████╗ ██████╗  ██████╗ ███╗   ███╗██████╗ ████████╗██╗      █████╗ ██████╗
██╔══██╗██╔══██╗██╔═══██╗████╗ ████║██╔══██╗╚══██╔══╝██║     ██╔══██╗██╔══██╗
██████╔╝██████╔╝██║   ██║██╔████╔██║██████╔╝   ██║   ██║     ███████║██████╔╝
██╔═══╝ ██╔══██╗██║   ██║██║╚██╔╝██║██╔═══╝    ██║   ██║     ██╔══██║██╔══██╗
██║     ██║  ██║╚██████╔╝██║ ╚═╝ ██║██║        ██║   ███████╗██║  ██║██████╔╝
╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝        ╚═╝   ╚══════╝╚═╝  ╚═╝╚═════╝
```

### ⚗️ Research-grade Prompt Evaluation Platform

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-promptlab--dev.streamlit.app-4f9eff?style=for-the-badge)](https://promptlab-dev.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Free_Tier-6366f1?style=for-the-badge)](https://openrouter.ai)

<br/>

> *"LLMs aren't unreliable. They're probabilistic. That's not a flaw — but it means evaluation isn't optional if you're building anything serious with them."*

<br/>

</div>

---

## 🧠 What is PromptLab?

PromptLab is a systematic LLM evaluation tool built to answer one question:

**How do you know if your prompt actually works — reliably?**

Most prompt testing happens in someone's head. *"This response feels better."* That's not engineering — that's guessing. PromptLab replaces gut feeling with measurable, reproducible data.

Built during a 10-week Prompt Engineering internship after observing that the same prompt, run 10 times, produced 10 structurally different answers — including one that hallucinated a citation.

---

## ✨ Features

### 🔁 Multi-Run Variability Analysis
Run the same prompt N times and measure how stable it is across runs. Outputs side-by-side with automatic outlier detection.

### 🌡️ Temperature Sweep
Test one prompt across multiple temperature settings (e.g. `0.0 → 1.5`) in a single click. See exactly where output shifts from deterministic to creative to chaotic.

### 🆚 Prompt Comparison Mode
Run Prompt A vs Prompt B under identical conditions. Side-by-side results with a full evaluation report.

### 📊 Determinism Index (DI)
A numerical stability score combining pairwise similarity and length variance.
```
DI = avg_similarity × (1 - coefficient_of_variation)
Range: 0–100  |  Higher = more stable
```
- **≥ 75** → Highly Stable 🟢
- **45–74** → Moderately Stable 🟡
- **< 45** → Unstable / Stochastic 🔴

### ⚠️ Hallucination Risk Detector
Rule-based heuristic that flags:
- Specific statistics without context
- Vague authority claims (`"studies show..."`, `"scientists say..."`)
- Absolute certainty language (`"proven"`, `"never fails"`)
- Fabricated citations or URLs

Returns **LOW / MEDIUM / HIGH** risk per run with explanation.

### 📐 Instruction Adherence Score
Did the model actually follow your constraints?
- Asked for one sentence — did it give one?
- Requested bullet points — are they there?
- Wanted JSON — is it valid?

Scored **0–100** per run.

### 🔍 Outlier Detection
Automatically flags runs that deviate significantly from others using similarity thresholds. Useful for catching edge cases before they hit production.

### 🤖 AI-Generated Research Summary
After each experiment, generates a concise analytical report covering variability observations, temperature effects, instruction-following quality, and recommended temperature range.

### 🔄 Reproducibility Mode
Save your exact experiment configuration as JSON. Re-upload it later to run the identical experiment again. Essential for comparing results over time.

### 📤 Export Ready
Export results as **JSON**, **CSV**, or **Markdown** report.

---

## 🏗️ Architecture

```
promptlab/
│
├── app.py                  # Main Streamlit app — UI + experiment orchestration
│
└── modules/
    ├── llm_client.py       # OpenRouter API client — calls, retries, token tracking
    ├── analysis.py         # All evaluation metrics — DI, similarity, outliers, heuristics
    ├── ai_summary.py       # AI-generated analytical summary
    ├── export.py           # JSON / CSV / Markdown export + config save/load
    └── templates.py        # Built-in prompt template library
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/devanshu651/PromptLab.git
cd PromptLab
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API key
Create a `.env` file:
```
OPENROUTER_API_KEY=your_key_here
```
Get a free key at [openrouter.ai](https://openrouter.ai) — no credit card required.

### 4. Run
```bash
streamlit run app.py
```

---

## 🧪 Experiment Modes

| Mode | Use Case |
|------|----------|
| 🔁 Multi-Run Variability | Test prompt stability across N runs |
| 🌡️ Temperature Sweep | Understand creativity vs determinism tradeoff |
| 🆚 Prompt Comparison | A/B test two prompts under identical conditions |

---

## 📚 Built-in Templates

| Template | Category |
|----------|----------|
| QA — Teacher vs Concise | Question Answering |
| Summarization — Paragraph vs Bullets | Summarization |
| Extraction — JSON vs Freeform | Data Extraction |
| Creative Writing — Constrained vs Free | Creative |
| Structured Reasoning — CoT vs Direct | Reasoning |
| Code — Documented vs Minimal | Code Generation |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| LLM API | OpenRouter (OpenAI-compatible) |
| Models | Free tier — DeepSeek, Qwen, Llama, Gemini |
| Metrics | TF-IDF cosine similarity (no external ML deps) |
| Export | JSON, CSV, Markdown |
| Deployment | Streamlit Cloud |

---

## 📈 Key Metrics Explained

```
Pairwise Similarity  →  TF-IDF cosine similarity across all run pairs
Determinism Index    →  avg_sim × (1 - CV)  — combined stability score
Hallucination Risk   →  Rule-based flag count → LOW / MEDIUM / HIGH
Instruction Score    →  Constraint compliance check → 0–100
Outlier Detection    →  Runs with avg similarity < threshold
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | ✅ | Your OpenRouter API key |

For Streamlit Cloud deployment, add this in **App Settings → Secrets**:
```toml
OPENROUTER_API_KEY = "your_key_here"
```

---

<div align="center">

<br/>

**Built by [Devanshu Dipak Raut](https://github.com/devanshu651)**

*10-week Prompt Engineering Internship — EduSkills Academy*

<br/>

[![Live Demo](https://img.shields.io/badge/Try_it_Live-promptlab--dev.streamlit.app-4f9eff?style=flat-square)](https://promptlab-dev.streamlit.app/)

<br/>

```
⚗️ PromptLab Pro v3.0.0
```

</div>