"""
modules/export.py
──────────────────
Export experiment results as JSON, CSV, or Markdown.
"""
from __future__ import annotations
import json, csv, io
from datetime import datetime


def to_json(experiment: dict) -> str:
    return json.dumps(experiment, indent=2, ensure_ascii=False)


def to_csv(experiment: dict) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Prompt", "Run", "Temperature", "Words", "Latency(ms)", "Response"])
    for entry in experiment.get("runs", []):
        writer.writerow([
            entry.get("prompt_label", ""),
            entry.get("run_index", ""),
            entry.get("temperature", ""),
            entry.get("word_count", ""),
            entry.get("latency_ms", ""),
            entry.get("text", "").replace("\n", " "),
        ])
    return output.getvalue()


def to_markdown(experiment: dict) -> str:
    meta = experiment.get("metadata", {})
    lines = [
        "# PromptLab Experiment Report",
        "",
        f"**Generated:** {meta.get('timestamp', datetime.now().isoformat())}",
        f"**Model:** `{meta.get('model', 'N/A')}`",
        f"**Temperature:** {meta.get('temperature', 'N/A')}",
        f"**Runs:** {meta.get('n_runs', 'N/A')}",
        "",
        "---",
        "",
    ]

    # Prompts
    for key in ["prompt_a", "prompt_b"]:
        if key in experiment:
            label = "Prompt A" if key == "prompt_a" else "Prompt B"
            lines += [f"## {label}", "", f"```", experiment[key], "```", ""]

    # Results
    for entry in experiment.get("runs", []):
        lines += [
            f"### {entry.get('prompt_label')} — Run {entry.get('run_index', '')+1}",
            "",
            f"*Words: {entry.get('word_count')} | Latency: {entry.get('latency_ms')}ms*",
            "",
            entry.get("text", ""),
            "",
            "---",
            "",
        ]

    # Metrics
    if "metrics" in experiment:
        lines += ["## Metrics Summary", ""]
        for k, v in experiment["metrics"].items():
            lines.append(f"- **{k}:** {v}")

    return "\n".join(lines)


def build_experiment_dict(
    prompt_a: str,
    prompt_b: str,
    task: str,
    model: str,
    temperature: float,
    n_runs: int,
    responses_a: list[dict],
    responses_b: list[dict],
    metrics_a: dict,
    metrics_b: dict,
    consistency_a: dict,
    consistency_b: dict,
) -> dict:
    """Assemble all experiment data into one serializable dict."""
    now = datetime.now().isoformat()
    runs = []

    for i, r in enumerate(responses_a):
        runs.append({
            "prompt_label": "Prompt A",
            "run_index": i,
            "temperature": temperature,
            "text": r.get("text", ""),
            "word_count": len(r.get("text", "").split()),
            "latency_ms": r.get("latency_ms", 0),
            "tokens_prompt": r.get("tokens_prompt", 0),
            "tokens_completion": r.get("tokens_completion", 0),
            "error": r.get("error"),
        })

    for i, r in enumerate(responses_b):
        runs.append({
            "prompt_label": "Prompt B",
            "run_index": i,
            "temperature": temperature,
            "text": r.get("text", ""),
            "word_count": len(r.get("text", "").split()),
            "latency_ms": r.get("latency_ms", 0),
            "tokens_prompt": r.get("tokens_prompt", 0),
            "tokens_completion": r.get("tokens_completion", 0),
            "error": r.get("error"),
        })

    return {
        "metadata": {
            "timestamp": now,
            "model": model,
            "temperature": temperature,
            "n_runs": n_runs,
            "task": task,
        },
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "runs": runs,
        "metrics": {
            "prompt_a_avg_words": metrics_a.get("mean"),
            "prompt_a_std_dev": metrics_a.get("std_dev"),
            "prompt_a_stability": metrics_a.get("stability"),
            "prompt_a_consistency": consistency_a.get("score"),
            "prompt_a_consistency_label": consistency_a.get("label"),
            "prompt_b_avg_words": metrics_b.get("mean"),
            "prompt_b_std_dev": metrics_b.get("std_dev"),
            "prompt_b_stability": metrics_b.get("stability"),
            "prompt_b_consistency": consistency_b.get("score"),
            "prompt_b_consistency_label": consistency_b.get("label"),
        },
    }