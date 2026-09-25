from __future__ import annotations

import re
from typing import Any


from database.chroma import ChromaCloudStore
from providers.harrier import HarrierEmbeddingProvider
from providers.reranker import HostedReranker
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.bm25 import BM25OkapiLite


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class HybridRetriever:
    """Cloud-native multilingual hybrid retriever.

    Dense: hosted Microsoft Harrier embeddings + Chroma Cloud.
    Sparse: BM25 over the active session/source corpus.
    Fusion: Reciprocal Rank Fusion (default), optionally followed by a hosted reranker.
    """

    def __init__(
        self,
        store: ChromaCloudStore,
        embedder: HarrierEmbeddingProvider,
        reranker: HostedReranker | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def index_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        texts = [str(chunk["text"]) for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            metadata = dict(chunk.get("metadata") or {})
            chunk_id = str(metadata.get("chunk_id") or f"chunk-{index}")
            ids.append(chunk_id)
            metadatas.append(metadata)
        self.store.upsert(ids, texts, metadatas, embeddings)
        return len(chunks)

    def vector_search(self, query: str, *, top_k: int = 15, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        vector = self.embedder.embed_query(query)
        results = self.store.vector_query(vector, top_k=top_k, where=where)
        # Chroma distances are lower-is-better; rank order is already correct.
        for rank, item in enumerate(results, start=1):
            item["vector_rank"] = rank
        return results

    def bm25_search(self, query: str, *, top_k: int = 15, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        corpus = self.store.get_documents(where=where)
        if not corpus:
            return []
        bm25 = BM25OkapiLite([_tokenize(doc["text"]) for doc in corpus])
        scores = bm25.get_scores(_tokenize(query))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: min(top_k, len(corpus))]
        output: list[dict[str, Any]] = []
        for rank, idx in enumerate(indices, start=1):
            doc = dict(corpus[idx])
            doc["bm25_score"] = float(scores[idx])
            doc["bm25_rank"] = rank
            output.append(doc)
        return output

    def get_corpus(self, *, where: dict[str, Any] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Return indexed chunks for document-wide/hierarchical processing."""
        return self.store.get_documents(where=where, limit=limit)

    def retrieve(
        self,
        query: str,
        *,
        where: dict[str, Any] | None = None,
        bm25_k: int = 15,
        vector_k: int = 15,
        final_k: int = 5,
    ) -> list[dict[str, Any]]:
        sparse = self.bm25_search(query, top_k=bm25_k, where=where)
        dense = self.vector_search(query, top_k=vector_k, where=where)
        fused = reciprocal_rank_fusion([sparse, dense])
        if self.reranker and fused:
            return self.reranker.rerank(query, fused, final_k)
        return fused[:final_k]
