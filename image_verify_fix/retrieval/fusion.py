from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int = 60,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = str(doc["id"])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            docs.setdefault(doc_id, doc)
    fused = []
    for doc_id, score in scores.items():
        item = dict(docs[doc_id])
        item["rrf_score"] = score
        fused.append(item)
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused
