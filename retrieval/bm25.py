from __future__ import annotations

import math
from collections import Counter


class BM25OkapiLite:
    """Small dependency-free BM25 implementation for deployment portability."""

    def __init__(self, corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        self.doc_freqs: Counter[str] = Counter()
        for doc in corpus:
            self.doc_freqs.update(set(doc))
        self.n = len(corpus)

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5)) if self.n else 0.0

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for doc, dl in zip(self.corpus, self.doc_lens):
            counts = Counter(doc)
            score = 0.0
            for term in query_tokens:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
                score += self._idf(term) * ((tf * (self.k1 + 1.0)) / denom)
            scores.append(score)
        return scores
