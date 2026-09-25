from __future__ import annotations

from pathlib import Path
from typing import Any

from app.schemas import Chunk, IngestionResult
from ingestion.chunker import StructureAwareChunker
from ingestion.router import IngestionRouter
from retrieval.hybrid_retrieval import HybridRetriever


class Phase12Pipeline:
    """Public façade for the completed Phase 1 + Phase 2 pipeline."""

    def __init__(self, router: IngestionRouter, chunker: StructureAwareChunker, retriever: HybridRetriever):
        self.router = router
        self.chunker = chunker
        self.retriever = retriever

    def ingest_and_index(self, path: str | Path, *, session_id: str) -> tuple[IngestionResult, list[Chunk]]:
        result = self.router.ingest(path)
        chunks = self.chunker.chunk(result, session_id=session_id)
        self.retriever.index_chunks([chunk.to_retrieval_dict() for chunk in chunks])
        return result, chunks

    def retrieve(
        self,
        query: str,
        *,
        session_id: str,
        source_id: str | None = None,
        final_k: int = 5,
    ) -> list[dict[str, Any]]:
        clauses: list[dict[str, Any]] = [{"session_id": session_id}]
        if source_id:
            clauses.append({"source_id": source_id})
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        return self.retriever.retrieve(query, where=where, final_k=final_k)
