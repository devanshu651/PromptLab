"""
modules/analysis.py
────────────────────
All evaluation logic:
  - Consistency scoring (TF-IDF cosine similarity)
  - Hallucination risk heuristics
  - JSON validation
  - Length statistics
  - Variability summary
"""
from __future__ import annotations
import json, re, math
from difflib import SequenceMatcher


# ── Consistency Scoring ──────────────────────────────────────────────────────

def _tfidf_cosine(a: str, b: str) -> float:
    """Lightweight TF-IDF cosine similarity — no external deps."""
    def tokenize(t): return re.findall(r'\w+', t.lower())
    def tfidf(tokens):
        freq: dict[str, int] = {}
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
        return freq

    ta, tb = tokenize(a), tokenize(b)
    fa, fb = tfidf(ta), tfidf(tb)
    vocab = set(fa) | set(fb)
    dot = sum(fa.get(w, 0) * fb.get(w, 0) for w in vocab)
    na = math.sqrt(sum(v**2 for v in fa.values()))
    nb = math.sqrt(sum(v**2 for v in fb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def consistency_score(responses: list[str]) -> dict:
    """
    Compute pairwise cosine similarity across all response pairs.
    Returns:
      - score: float 0–100
      - label: str
      - pairs: list of (i, j, sim)
    """
    if len(responses) < 2:
        return {"score": 100.0, "label": "High Consistency 🟢", "pairs": []}

    pairs = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = _tfidf_cosine(responses[i], responses[j])
            pairs.append((i, j, round(sim * 100, 1)))

    avg = sum(p[2] for p in pairs) / len(pairs)

    if avg >= 70:
        label = "High Consistency 🟢"
    elif avg >= 40:
        label = "Moderate Consistency 🟡"
    else:
        label = "Low Consistency 🔴"

    return {"score": round(avg, 1), "label": label, "pairs": pairs}


# ── Hallucination Risk ───────────────────────────────────────────────────────

_CITATION_PATTERNS = [
    r'\b(19|20)\d{2}\b',                          # years like 2019, 2023
    r'\b\d+(\.\d+)?%',                             # percentages
    r'\b(according to|study shows|research shows|scientists found|experts say)',
    r'\b(et al\.?|doi:|arxiv:|pubmed:)',           # academic citation markers
    r'\b\d{1,3}(,\d{3})+\b',                      # large numbers like 1,234,567
    r'https?://\S+',                               # URLs
    r'\b(journal of|proceedings of|published in)', # publication refs
]

_HIGH_RISK_PHRASES = [
    "as of 2024", "as of 2025", "as of 2026",
    "recent study", "new research", "latest data",
    "officially", "proven", "confirmed by",
    "100%", "never fails", "always works",
]


def hallucination_risk(text: str) -> dict:
    """
    Rule-based hallucination risk heuristic.
    Returns:
      - level: Low | Medium | High
      - score: int 0–100
      - flags: list of flagged snippets
    """
    flags = []
    score = 0

    for pat in _CITATION_PATTERNS:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            flags.append(f"Pattern `{pat}` → {len(matches)} match(es)")
            score += len(matches) * 5

    for phrase in _HIGH_RISK_PHRASES:
        if phrase.lower() in text.lower():
            flags.append(f"High-risk phrase: '{phrase}'")
            score += 15

    # Long responses with many specific claims tend to be riskier
    word_count = len(text.split())
    if word_count > 400:
        score += 10
        flags.append("Long response (>400 words) — more surface area for errors")

    score = min(score, 100)

    if score < 25:
        level = "Low Risk 🟢"
    elif score < 55:
        level = "Medium Risk 🟡"
    else:
        level = "High Risk 🔴"

    return {"level": level, "score": score, "flags": flags}


# ── JSON Validation ──────────────────────────────────────────────────────────

def validate_json(text: str) -> dict:
    """Try to parse text as JSON. Returns pass/fail + parsed object."""
    # Strip markdown code fences if present
    clean = re.sub(r'^```(json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r'\s*```$', '', clean)
    try:
        parsed = json.loads(clean)
        return {"valid": True, "parsed": parsed, "error": None}
    except json.JSONDecodeError as e:
        return {"valid": False, "parsed": None, "error": str(e)}


# ── Length Stats ─────────────────────────────────────────────────────────────

def length_stats(responses: list[str]) -> dict:
    lengths = [len(r.split()) for r in responses]
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((x - mean) ** 2 for x in lengths) / n
    std_dev = math.sqrt(variance)
    cv = std_dev / mean if mean > 0 else 0

    if cv < 0.05:
        stability = "Very Stable 🟢"
    elif cv < 0.15:
        stability = "Stable 🟡"
    elif cv < 0.30:
        stability = "Variable 🟠"
    else:
        stability = "Highly Variable 🔴"

    return {
        "lengths": lengths,
        "mean": round(mean, 1),
        "std_dev": round(std_dev, 2),
        "variance": round(variance, 1),
        "min": min(lengths),
        "max": max(lengths),
        "range": max(lengths) - min(lengths),
        "cv": round(cv, 3),
        "stability": stability,
    }


# ── Diff Highlighting ────────────────────────────────────────────────────────

def word_diff(a: str, b: str) -> list[dict]:
    """
    Returns list of {text, type} where type is 'equal'|'added'|'removed'.
    """
    words_a = a.split()
    words_b = b.split()
    matcher = SequenceMatcher(None, words_a, words_b)
    result = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            result.append({"text": " ".join(words_a[i1:i2]), "type": "equal"})
        elif op == 'insert':
            result.append({"text": " ".join(words_b[j1:j2]), "type": "added"})
        elif op == 'delete':
            result.append({"text": " ".join(words_a[i1:i2]), "type": "removed"})
        elif op == 'replace':
            result.append({"text": " ".join(words_a[i1:i2]), "type": "removed"})
            result.append({"text": " ".join(words_b[j1:j2]), "type": "added"})
    return result


# ── Prompt Comparison Summary ────────────────────────────────────────────────

def compare_prompts(
    responses_a: list[str],
    responses_b: list[str],
    label_a: str = "Prompt A",
    label_b: str = "Prompt B",
) -> str:
    """Generate a text summary comparing two prompt results."""
    stats_a = length_stats(responses_a)
    stats_b = length_stats(responses_b)
    cons_a = consistency_score(responses_a)
    cons_b = consistency_score(responses_b)

    winner_consistency = label_a if cons_a["score"] >= cons_b["score"] else label_b
    winner_length = label_a if stats_a["mean"] >= stats_b["mean"] else label_b

    lines = [
        f"**{label_a}** avg {stats_a['mean']} words | consistency {cons_a['score']}% ({cons_a['label']})",
        f"**{label_b}** avg {stats_b['mean']} words | consistency {cons_b['score']}% ({cons_b['label']})",
        "",
        f"→ **More consistent:** {winner_consistency}",
        f"→ **More detailed:** {winner_length}",
    ]
    return "\n".join(lines)