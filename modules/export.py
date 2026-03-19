"""
modules/export.py  — v3
────────────────────────
Export results as JSON / CSV / Markdown.
Save & load experiment configurations for reproducibility.
"""
from __future__ import annotations
import json, csv, io
from datetime import datetime

APP_VERSION = "3.0.0"


# ── Config save/load ─────────────────────────────────────────────────────────

def save_config(
    prompt_a: str,
    prompt_b: str,
    task: str,
    model: str,
    temperature: float,
    n_runs: int,
    max_tokens: int,
    temperatures: list[float],
    json_mode: bool,
    mode: str,
) -> str:
    """Serialize experiment config to JSON string."""
    config = {
        "app_version" : APP_VERSION,
        "timestamp"   : datetime.now().isoformat(),
        "mode"        : mode,
        "task"        : task,
        "prompt_a"    : prompt_a,
        "prompt_b"    : prompt_b,
        "model"       : model,
        "temperature" : temperature,
        "temperatures": temperatures,
        "n_runs"      : n_runs,
        "max_tokens"  : max_tokens,
        "json_mode"   : json_mode,
    }
    return json.dumps(config, indent=2)


def load_config(json_str: str) -> dict:
    """Deserialize config from JSON string. Returns dict or raises ValueError."""
    try:
        cfg = json.loads(json_str)
        required = ["prompt_a", "model", "temperature", "n_runs"]
        for key in required:
            if key not in cfg:
                raise ValueError(f"Missing required key: {key}")
        return cfg
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


# ── Result exports ────────────────────────────────────────────────────────────

def to_json(experiment: dict) -> str:
    return json.dumps(experiment, indent=2, ensure_ascii=False)


def to_csv(experiment: dict) -> str:
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["prompt_label", "run", "temperature", "words",
                "latency_ms", "tokens_completion", "error", "text"])
    for r in experiment.get("runs", []):
        w.writerow([
            r.get("prompt_label", ""),
            r.get("run_index", ""),
            r.get("temperature", ""),
            r.get("word_count", ""),
            r.get("latency_ms", ""),
            r.get("tokens_completion", ""),
            r.get("error", ""),
            r.get("text", "").replace("\n", " "),
        ])
    return output.getvalue()


def to_markdown(experiment: dict) -> str:
    meta = experiment.get("metadata", {})
    lines = [
        "# PromptLab Pro — Experiment Report",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| **Generated** | {meta.get('timestamp', '')} |",
        f"| **Model** | `{meta.get('model', '')}` |",
        f"| **Temperature** | {meta.get('temperature', '')} |",
        f"| **Runs** | {meta.get('n_runs', '')} |",
        f"| **Mode** | {meta.get('mode', '')} |",
        f"| **App Version** | {meta.get('app_version', APP_VERSION)} |",
        "",
        "---",
        "",
    ]

    for key in ["prompt_a", "prompt_b"]:
        if experiment.get(key):
            label = "Prompt A" if key == "prompt_a" else "Prompt B"
            lines += [f"## {label}", "", "```", experiment[key], "```", ""]

    lines += ["## Raw Outputs", ""]
    for r in experiment.get("runs", []):
        lines += [
            f"### {r.get('prompt_label')} — Run {r.get('run_index', 0)+1}",
            f"*{r.get('word_count')} words · {r.get('latency_ms')}ms*",
            "",
            r.get("text", ""),
            "",
            "---",
            "",
        ]

    if experiment.get("ai_summary"):
        lines += ["## AI-Generated Analysis", "", experiment["ai_summary"], ""]

    if "metrics" in experiment:
        lines += ["## Metrics", ""]
        for k, v in experiment["metrics"].items():
            lines.append(f"- **{k}:** {v}")

    return "\n".join(lines)


def build_experiment_dict(
    prompt_a, prompt_b, task, model, temperature,
    n_runs, responses_a, responses_b,
    metrics_a, metrics_b, di_a, di_b,
    ai_summary: str = "",
    mode: str = "",
) -> dict:
    runs = []
    for i, r in enumerate(responses_a):
        runs.append({
            "prompt_label"      : "Prompt A",
            "run_index"         : i,
            "temperature"       : temperature,
            "text"              : r.get("text", ""),
            "word_count"        : len(r.get("text", "").split()),
            "latency_ms"        : r.get("latency_ms", 0),
            "tokens_prompt"     : r.get("tokens_prompt", 0),
            "tokens_completion" : r.get("tokens_completion", 0),
            "error"             : r.get("error"),
        })
    for i, r in enumerate(responses_b):
        runs.append({
            "prompt_label"      : "Prompt B",
            "run_index"         : i,
            "temperature"       : temperature,
            "text"              : r.get("text", ""),
            "word_count"        : len(r.get("text", "").split()),
            "latency_ms"        : r.get("latency_ms", 0),
            "tokens_prompt"     : r.get("tokens_prompt", 0),
            "tokens_completion" : r.get("tokens_completion", 0),
            "error"             : r.get("error"),
        })

    return {
        "metadata": {
            "timestamp"  : datetime.now().isoformat(),
            "model"      : model,
            "temperature": temperature,
            "n_runs"     : n_runs,
            "task"       : task,
            "mode"       : mode,
            "app_version": APP_VERSION,
        },
        "prompt_a"  : prompt_a,
        "prompt_b"  : prompt_b,
        "runs"      : runs,
        "ai_summary": ai_summary,
        "metrics": {
            "A_avg_words"        : metrics_a.get("mean"),
            "A_std_dev"          : metrics_a.get("std_dev"),
            "A_determinism_index": di_a.get("di"),
            "A_stability"        : di_a.get("label"),
            "B_avg_words"        : metrics_b.get("mean"),
            "B_std_dev"          : metrics_b.get("std_dev"),
            "B_determinism_index": di_b.get("di"),
            "B_stability"        : di_b.get("label"),
        },
    }