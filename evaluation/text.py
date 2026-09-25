from __future__ import annotations
import re
from collections import Counter

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def normalized_exact_match(prediction: str, reference: str) -> float:
    return 1.0 if " ".join(tokens(prediction)) == " ".join(tokens(reference)) else 0.0


def token_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return (1.0, 1.0, 1.0) if p == r else (0.0, 0.0, 0.0)
    overlap = sum((Counter(p) & Counter(r)).values())
    precision = overlap / len(p)
    recall = overlap / len(r)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _lcs_len(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> tuple[float, float, float]:
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return (1.0, 1.0, 1.0) if p == r else (0.0, 0.0, 0.0)
    lcs = _lcs_len(p, r)
    precision, recall = lcs / len(p), lcs / len(r)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def reference_metrics(prediction: str, reference: str) -> dict[str, float]:
    tp, tr, tf = token_f1(prediction, reference)
    rp, rr, rf = rouge_l(prediction, reference)
    ref_len = max(1, len(tokens(reference)))
    return {
        "exact_match": normalized_exact_match(prediction, reference),
        "token_precision": tp,
        "token_recall": tr,
        "token_f1": tf,
        "rouge_l_precision": rp,
        "rouge_l_recall": rr,
        "rouge_l_f1": rf,
        "length_ratio": len(tokens(prediction)) / ref_len,
    }
