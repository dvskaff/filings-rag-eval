"""Automatic grading of answers against FinanceBench gold answers: Correct / Incorrect / Refusal.

FinanceBench's published results come with human labels, so this grader is checked against them (agreement is
reported on the dashboard) instead of being trusted blindly.
"""

from __future__ import annotations

import re

REFUSAL = re.compile(
    r"(i don't know|i do not know|not (?:be )?(?:possible|able) to (?:determine|calculate|answer)|cannot (?:be )?"
    r"(?:determine|calculate|answer|provide|find)|can't (?:determine|calculate|answer)|unable to (?:determine|"
    r"calculate|answer|provide|find)|(?:is|are) not (?:provided|included|available|mentioned|disclosed|specified)|"
    r"does not (?:provide|include|contain|mention|specify|disclose)|do not (?:provide|include|contain|have)|"
    r"no (?:information|data) (?:is )?(?:provided|available)|insufficient information|not enough information|"
    r"no (?:direct|specific|explicit) (?:information|mention|figure|data)|not (?:directly |explicitly |specifically )?"
    r"(?:stated|mentioned|reported|shown|included|provided)|would need|i'm sorry|i am sorry|unfortunately)",
    re.I,
)
NUM = re.compile(r"(?<![\w.])[-−(]?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|percent|billion|bn|million|mn|mm|m\b|thousand|k\b)?",
                 re.I)  # fmt: skip
NOISE = re.compile(
    r"\[\d+(?:\s*,\s*\d+)*\]|\b\d{1,2}-[KQ]\b|\bQ[1-4]\b|\bFY\d{2,4}\b", re.I
)  # citations, form names, periods
SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
STOP = set(
    "the a an of in to for and or is are was were be by on at as with that this its it from than which".split()
)


def numbers(text: str) -> list[tuple[float, str]]:
    """Every number in the text with its unit word ('' if none). Years (1990-2039 written plainly, without a unit,
    $ or thousands separator) are skipped."""
    out = []
    text = NOISE.sub(" ", str(text).replace("−", "-"))
    for m in NUM.finditer(text):
        raw = m.group(1).replace(",", "") + (m.group(2) or "")
        unit = (m.group(3) or "").lower()
        v = float(raw)
        plain = "$" not in m.group(0) and "," not in m.group(1)
        if not unit and m.group(2) is None and plain and 1990 <= v <= 2039:
            continue
        out.append((v, unit))
    return out


def _variants(v: float, unit: str) -> set[float]:
    """A number can be written in different units (1,577 million = 1.577 billion); compare on several scales."""
    if unit in ("%", "percent"):
        return {v, v / 100}
    if unit not in SCALE:
        return {v}  # a bare number is compared as written
    base = v * SCALE[unit]
    return {v, base, base / 1e3, base / 1e6, base / 1e9}


def close(a: float, b: float, tol: float = 0.015) -> bool:
    """Relative tolerance (answers are often rounded: 0.96 vs 0.955, 1,577 vs 1,577.0)."""
    return abs(a - b) <= tol * abs(b) if b else abs(a) < 1e-9


def answer_kind(gold: str) -> str:
    g = str(gold).strip().lower()
    if re.match(r"^(yes|no)\b", g):
        return "yes/no"
    nums = numbers(gold)
    if nums and len(g) <= 60:
        return "numeric"
    return "text"


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in STOP and len(t) > 1}


NEGATIVE = re.compile(
    r"\b(not|no|never|isn't|doesn't|don't|didn't|wasn't|aren't|declin\w*|decreas\w*|deteriorat\w*|"
    r"weak\w*|worse\w*|lower|fell|unhealthy|insufficient|limited)\b",
    re.I,
)


def polarity(answer: str) -> str | None:
    """'yes' / 'no' from an explicit Yes/No, otherwise from negative wording in the first sentence."""
    head = answer.strip().lower()
    explicit = re.match(r"^\W*(yes|no)\b", head)
    if explicit:
        return explicit.group(1)
    first = re.split(r"(?<=[.!?])\s", head, maxsplit=1)[0]
    found = re.findall(r"\b(yes|no)\b", first)
    if found:
        return found[0]
    return "no" if NEGATIVE.search(first) else "yes"


def grade(answer: str, gold, kind: str | None = None, question: str = "") -> str:
    """'Correct', 'Incorrect' or 'Refusal'. With the question, words the gold answer merely repeats from it don't count
    as evidence for free-text answers (otherwise restating the question would look correct)."""
    a = str(answer or "").strip()
    kind = kind or answer_kind(str(gold))
    if not a or REFUSAL.search(a):
        return "Refusal"
    return "Correct" if _has_match(a, str(gold), kind, question) else "Incorrect"


def _has_match(a: str, gold: str, kind: str, question: str = "") -> bool:
    if kind == "numeric":
        g = numbers(gold)[0]
        targets = _variants(*g)
        return any(close(x, t) for v, u in numbers(a) for x in _variants(v, u) for t in targets)
    if kind == "yes/no":
        return polarity(a) == gold.strip().lower().split()[0].strip(".,")
    gt, at = tokens(gold), tokens(a)
    gt = (gt - tokens(question)) or gt  # the key terms the question doesn't already give away
    if not gt:
        return False
    recall = len(gt & at) / len(gt)
    gnums = numbers(gold)
    if gnums:  # text answers that hinge on a figure: the figure must be there too
        ok = any(close(x, t) for v, u in numbers(a) for x in _variants(v, u) for t in _variants(*gnums[0]))
        return ok and recall >= 0.3
    return recall >= 0.5
