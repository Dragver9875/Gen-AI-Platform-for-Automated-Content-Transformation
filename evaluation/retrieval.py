from __future__ import annotations
import math


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(x) for x in items))


def precision_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = set(relevant)
    return sum(1 for item in ranked[:k] if item in rel) / float(k)


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel.intersection(ranked[:k])) / float(len(rel))


def hit_rate_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    return 1.0 if any(item in rel for item in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for i, item in enumerate(ranked, 1):
        if item in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    if not rel or k <= 0:
        return 0.0
    dcg = sum((1.0 / math.log2(i + 2)) for i, item in enumerate(ranked[:k]) if item in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(ranked: list[str], relevant: list[str], k: int) -> dict[str, float]:
    ranked = _unique(ranked)
    relevant = _unique(relevant)
    return {
        f"precision@{k}": precision_at_k(ranked, relevant, k),
        f"recall@{k}": recall_at_k(ranked, relevant, k),
        f"hit_rate@{k}": hit_rate_at_k(ranked, relevant, k),
        "mrr": reciprocal_rank(ranked, relevant),
        f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
    }
