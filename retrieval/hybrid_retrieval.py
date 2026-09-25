from __future__ import annotations

import re
from typing import Any


from database.chroma import ChromaCloudStore
from providers.harrier import HarrierEmbeddingProvider
from providers.reranker import HostedReranker
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.bm25 import BM25OkapiLite
from retrieval.config import RetrievalConfig


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
        default_config: RetrievalConfig | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.default_config = default_config or RetrievalConfig()

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
        bm25_k: int | None = None,
        vector_k: int | None = None,
        final_k: int | None = None,
        config: RetrievalConfig | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cfg = (config if isinstance(config, RetrievalConfig) else RetrievalConfig.model_validate(config or self.default_config.model_dump()))
        if bm25_k is not None:
            cfg = cfg.model_copy(update={"bm25_k": bm25_k})
        if vector_k is not None:
            cfg = cfg.model_copy(update={"vector_k": vector_k})
        if final_k is not None:
            cfg = cfg.model_copy(update={"final_k": final_k})

        ranked: list[list[dict[str, Any]]] = []
        if cfg.strategy in {"hybrid", "lexical"}:
            ranked.append(self.bm25_search(query, top_k=cfg.bm25_k, where=where))
        if cfg.strategy in {"hybrid", "dense"}:
            ranked.append(self.vector_search(query, top_k=cfg.vector_k, where=where))
        if not ranked:
            return []
        fused = ranked[0] if len(ranked) == 1 else reciprocal_rank_fusion(ranked, k=cfg.rrf_k)
        if self.reranker and cfg.use_reranker and fused:
            return self.reranker.rerank(query, fused, cfg.final_k)
        return fused[: cfg.final_k]
