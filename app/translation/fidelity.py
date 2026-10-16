"""Cheap, model-free checks that a translated chunk did not drop or invent content."""

import re
import unicodedata
from collections import Counter

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def chrf(hypothesis: str, reference: str, *, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, Popović 2015), whitespace removed, in [0, 1]."""
    hyp = re.sub(r"\s+", "", hypothesis)
    ref = re.sub(r"\s+", "", reference)
    if not hyp or not ref:
        return 1.0 if hyp == ref else 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        h = Counter(hyp[i : i + n] for i in range(len(hyp) - n + 1))
        r = Counter(ref[i : i + n] for i in range(len(ref) - n + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def _ascii_digits(number: str) -> str:
    """'४२' (Devanagari) or '۴۲' (Urdu) -> '42', so a translation may switch digit systems."""
    return "".join(str(unicodedata.decimal(c)) if c.isdigit() else c for c in number)


def numbers(text: str) -> Counter[str]:
    text = text.replace("\u066b", ".").replace("\u066c", ",")  # Arabic decimal / thousands separators
    return Counter(_ascii_digits(m).replace(",", "") for m in _NUM_RE.findall(text))


def check_chunk(source: str, output: str, *, same_language: bool, min_chrf: float) -> list[str]:
    """Return a list of problems; an empty list means the chunk passed."""
    problems: list[str] = []
    if not output.strip():
        return ["empty output"]

    ratio = len(output) / max(1, len(source))
    low, high = (0.75, 1.35) if same_language else (0.4, 2.5)
    if not low <= ratio <= high:
        problems.append(f"length ratio {ratio:.2f} outside [{low}, {high}]")

    missing = numbers(source) - numbers(output)
    if missing:
        problems.append(f"numbers missing: {sorted(missing)[:5]}")

    if same_language:
        score = chrf(output, source)
        if score < min_chrf:
            problems.append(f"chrF {score:.2f} < {min_chrf}")
    return problems
