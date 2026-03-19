"""
modules/analysis.py  — v3
──────────────────────────
Research-grade evaluation metrics:
  - Pairwise TF-IDF cosine similarity
  - Determinism Index
  - Outlier detection
  - Hallucination risk heuristics
  - JSON schema compliance
  - Length variance stats
  - Instruction adherence heuristic
"""
from __future__ import annotations
import json, re, math
from difflib import SequenceMatcher


# ══════════════════════════════════════════════════════════════════════════════
# TEXT SIMILARITY  (TF-IDF cosine, no external deps)
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())

def _term_freq(tokens: list[str]) -> dict[str, float]:
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    n = len(tokens) or 1
    return {t: c / n for t, c in freq.items()}

def _cosine(fa: dict, fb: dict) -> float:
    vocab = set(fa) | set(fb)
    dot  = sum(fa.get(w, 0) * fb.get(w, 0) for w in vocab)
    na   = math.sqrt(sum(v**2 for v in fa.values()))
    nb   = math.sqrt(sum(v**2 for v in fb.values()))
    return dot / (na * nb) if na and nb else 0.0

def pairwise_similarity(texts: list[str]) -> dict:
    """
    Returns:
      avg_sim  : float  0-100
      pairs    : list[(i, j, sim_pct)]
      matrix   : list[list[float]]  n×n
    """
    n = len(texts)
    tfs = [_term_freq(_tokenize(t)) for t in texts]
    matrix = [[0.0]*n for _ in range(n)]
    pairs  = []
    for i in range(n):
        matrix[i][i] = 100.0
        for j in range(i+1, n):
            s = round(_cosine(tfs[i], tfs[j]) * 100, 1)
            matrix[i][j] = matrix[j][i] = s
            pairs.append((i, j, s))
    avg = round(sum(p[2] for p in pairs) / len(pairs), 1) if pairs else 100.0
    return {"avg_sim": avg, "pairs": pairs, "matrix": matrix}


# ══════════════════════════════════════════════════════════════════════════════
# LENGTH STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def length_stats(texts: list[str]) -> dict:
    lengths = [len(t.split()) for t in texts]
    n = len(lengths) or 1
    mean = sum(lengths) / n
    variance = sum((x - mean)**2 for x in lengths) / n
    std_dev  = math.sqrt(variance)
    cv = std_dev / mean if mean else 0
    return {
        "lengths" : lengths,
        "mean"    : round(mean, 1),
        "std_dev" : round(std_dev, 2),
        "variance": round(variance, 1),
        "cv"      : round(cv, 3),
        "min"     : min(lengths),
        "max"     : max(lengths),
        "range"   : max(lengths) - min(lengths),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DETERMINISM INDEX
# ══════════════════════════════════════════════════════════════════════════════

def determinism_index(avg_sim: float, cv: float) -> dict:
    """
    DI = avg_similarity × (1 - coefficient_of_variation)
    Range: 0–100.  Higher = more deterministic/stable.
    """
    di = avg_sim * max(0, 1 - cv)
    di = round(di, 1)
    if di >= 75:
        label = "Highly Stable 🟢"
        interp = "Outputs are nearly identical across runs. Low temperature or highly constrained prompt."
    elif di >= 45:
        label = "Moderately Stable 🟡"
        interp = "Outputs share core content but vary in phrasing and structure."
    else:
        label = "Unstable / Stochastic 🔴"
        interp = "High divergence between runs. Consider lowering temperature or constraining the prompt."
    return {"di": di, "label": label, "interpretation": interp}


# ══════════════════════════════════════════════════════════════════════════════
# OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_outliers(texts: list[str], threshold: float = 50.0) -> list[int]:
    """
    Returns indices of runs whose avg similarity to all others < threshold.
    """
    if len(texts) < 3:
        return []
    n = len(texts)
    tfs = [_term_freq(_tokenize(t)) for t in texts]
    outliers = []
    for i in range(n):
        sims = [_cosine(tfs[i], tfs[j]) * 100 for j in range(n) if j != i]
        avg  = sum(sims) / len(sims)
        if avg < threshold:
            outliers.append(i)
    return outliers


# ══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION ADHERENCE HEURISTIC
# ══════════════════════════════════════════════════════════════════════════════

def instruction_adherence(prompt: str, response: str) -> dict:
    """
    Rule-based heuristic. Checks if response respects obvious constraints
    in the prompt (e.g. 'one sentence', 'bullet points', 'JSON', 'step by step').
    Returns score 0-100 and list of checks.
    """
    checks = []
    score  = 100

    p_lower = prompt.lower()
    r_lower = response.lower()

    # Length constraints
    if "one sentence" in p_lower or "single sentence" in p_lower:
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s for s in sentences if s.strip()]
        passed = len(sentences) <= 2
        checks.append({"rule": "One sentence", "passed": passed})
        if not passed: score -= 25

    if "one paragraph" in p_lower or "single paragraph" in p_lower:
        paras = [p for p in response.split('\n\n') if p.strip()]
        passed = len(paras) <= 1
        checks.append({"rule": "One paragraph", "passed": passed})
        if not passed: score -= 20

    # Format constraints
    if "bullet" in p_lower or "list" in p_lower:
        passed = bool(re.search(r'[-•*]\s|\d+\.\s', response))
        checks.append({"rule": "Bullet/list format", "passed": passed})
        if not passed: score -= 20

    if "step by step" in p_lower:
        passed = bool(re.search(r'step\s*\d|^\d+[\.\)]\s', r_lower, re.MULTILINE))
        checks.append({"rule": "Step-by-step format", "passed": passed})
        if not passed: score -= 15

    if "json" in p_lower:
        try:
            clean = re.sub(r'^```(json)?\s*|\s*```$', '', response.strip(), flags=re.IGNORECASE)
            json.loads(clean)
            checks.append({"rule": "JSON format", "passed": True})
        except:
            checks.append({"rule": "JSON format", "passed": False})
            score -= 30

    if "concise" in p_lower or "brief" in p_lower or "short" in p_lower:
        word_count = len(response.split())
        passed = word_count <= 150
        checks.append({"rule": f"Concise (<150 words, got {word_count})", "passed": passed})
        if not passed: score -= 10

    if not checks:
        checks.append({"rule": "No explicit constraints detected", "passed": True})

    return {"score": max(0, score), "checks": checks}


# ══════════════════════════════════════════════════════════════════════════════
# HALLUCINATION RISK
# ══════════════════════════════════════════════════════════════════════════════

_CITATION_RE = [
    (r'\b(19|20)\d{2}\b',                          "Year reference"),
    (r'\b\d+(\.\d+)?%',                             "Percentage claim"),
    (r'\b(according to|study shows|research (shows|found|suggests))', "Unverified research claim"),
    (r'\b(et al\.?|doi:|arxiv:|pubmed:)',            "Academic citation marker"),
    (r'\b\d{1,3}(,\d{3})+\b',                       "Large specific number"),
    (r'https?://\S+',                                "URL / external link"),
    (r'\b(journal of|proceedings of|published in)',  "Publication reference"),
]

_HIGH_RISK = [
    ("as of 202", "Temporal claim (may be outdated/hallucinated)"),
    ("recent study", "Vague recent study reference"),
    ("proven", "Absolute certainty claim"),
    ("100%", "Absolute percentage"),
    ("never fails", "Absolute guarantee"),
    ("officially confirmed", "Unverifiable official claim"),
    ("scientists say", "Vague authority appeal"),
]

def hallucination_risk(text: str) -> dict:
    flags  = []
    score  = 0

    for pat, label in _CITATION_RE:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            flags.append(f"[{label}] — {len(matches)} occurrence(s)")
            score += len(matches) * 6

    for phrase, label in _HIGH_RISK:
        if phrase.lower() in text.lower():
            flags.append(f"[{label}] — phrase: '{phrase}'")
            score += 18

    wc = len(text.split())
    if wc > 400:
        score += 8
        flags.append(f"[Long response] — {wc} words, more exposure to errors")

    score = min(score, 100)
    level = "LOW 🟢" if score < 25 else ("MEDIUM 🟡" if score < 55 else "HIGH 🔴")
    return {"level": level, "score": score, "flags": flags}


# ══════════════════════════════════════════════════════════════════════════════
# JSON COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════════

def validate_json(text: str, expected_keys: list[str] | None = None) -> dict:
    clean = re.sub(r'^```(json)?\s*|\s*```$', '', text.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(clean)
        missing = []
        if expected_keys:
            missing = [k for k in expected_keys if k not in parsed]
        return {
            "valid"  : True,
            "parsed" : parsed,
            "missing": missing,
            "error"  : None,
            "pass"   : len(missing) == 0,
        }
    except json.JSONDecodeError as e:
        return {"valid": False, "parsed": None, "missing": [], "error": str(e), "pass": False}


# ══════════════════════════════════════════════════════════════════════════════
# WORD DIFF
# ══════════════════════════════════════════════════════════════════════════════

def word_diff(a: str, b: str) -> list[dict]:
    wa, wb = a.split(), b.split()
    m = SequenceMatcher(None, wa, wb)
    out = []
    for op, i1, i2, j1, j2 in m.get_opcodes():
        if op == 'equal':
            out.append({"text": " ".join(wa[i1:i2]), "type": "equal"})
        elif op == 'insert':
            out.append({"text": " ".join(wb[j1:j2]), "type": "added"})
        elif op == 'delete':
            out.append({"text": " ".join(wa[i1:i2]), "type": "removed"})
        elif op == 'replace':
            out.append({"text": " ".join(wa[i1:i2]), "type": "removed"})
            out.append({"text": " ".join(wb[j1:j2]), "type": "added"})
    return out