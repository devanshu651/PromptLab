"""
modules/ai_summary.py  — v3
────────────────────────────
Uses the LLM itself to generate a concise analytical research note
about the experiment results.
"""
from __future__ import annotations
from openai import OpenAI


SUMMARY_SYSTEM = """You are a senior NLP evaluation researcher writing a concise technical report.
Your audience is developers and ML engineers — NOT beginners.
Be precise, analytical, and neutral. Avoid marketing language.
Write in short paragraphs. Use bullet points only for lists of findings.
Keep the total report under 300 words."""


def build_summary_prompt(
    mode: str,
    task: str,
    prompt_a: str,
    prompt_b: str | None,
    texts_a: list[str],
    texts_b: list[str],
    metrics_a: dict,
    metrics_b: dict,
    di_a: dict,
    di_b: dict,
    temperatures: list[float],
    model: str,
) -> str:
    lines = [
        f"EXPERIMENT MODE: {mode}",
        f"MODEL: {model}",
        f"TASK: {task}",
        f"TEMPERATURE(S): {temperatures}",
        "",
        "=== PROMPT A ===",
        prompt_a,
        "",
        f"Runs: {len(texts_a)}",
        f"Avg words: {metrics_a.get('mean', 'N/A')}",
        f"Std dev: {metrics_a.get('std_dev', 'N/A')}",
        f"Determinism Index: {di_a.get('di', 'N/A')} — {di_a.get('label', '')}",
        "",
    ]

    for i, t in enumerate(texts_a[:3]):  # max 3 samples to avoid token overflow
        lines.append(f"[Run {i+1}]: {t[:300]}...")

    if prompt_b and texts_b:
        lines += [
            "",
            "=== PROMPT B ===",
            prompt_b,
            "",
            f"Runs: {len(texts_b)}",
            f"Avg words: {metrics_b.get('mean', 'N/A')}",
            f"Std dev: {metrics_b.get('std_dev', 'N/A')}",
            f"Determinism Index: {di_b.get('di', 'N/A')} — {di_b.get('label', '')}",
            "",
        ]
        for i, t in enumerate(texts_b[:3]):
            lines.append(f"[Run {i+1}]: {t[:300]}...")

    lines += [
        "",
        "=== YOUR TASK ===",
        "Write a concise analytical research note covering:",
        "1. Key variability observations across runs",
        "2. Temperature effects on output quality (if multiple temps tested)",
        "3. Instruction-following quality assessment",
        "4. Any outliers or anomalies detected",
        "5. Recommended temperature range for this task",
        "6. If two prompts: which performs better and why (be specific)",
    ]

    return "\n".join(lines)


def generate_summary(
    client: OpenAI,
    model: str,
    **kwargs,
) -> str:
    """
    Call LLM to generate analytical summary.
    Uses a fast model regardless of user selection to save tokens.
    """
    prompt = build_summary_prompt(model=model, **kwargs)

    try:
        resp = client.chat.completions.create(
            model="openrouter/free",  # Use fast/free model for meta-analysis
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,  # Low temp for analytical writing
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Summary generation failed: {e}]"