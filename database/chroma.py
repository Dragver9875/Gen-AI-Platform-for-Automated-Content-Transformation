from __future__ import annotations

import json
from typing import Any



def _safe_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    safe: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return safe


class ChromaCloudStore:
    def __init__(self, api_key: str, tenant: str, database: str, collection_name: str):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required for Chroma Cloud. Install requirements.txt") from exc
        self.client = chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=[_safe_metadata(m) for m in metadatas],
            embeddings=embeddings,
        )

    def vector_query(self, query_embedding: list[float], *, top_k: int, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        result = self.collection.query(**kwargs)
        return [
            {"id": idx, "text": text, "metadata": metadata or {}, "distance": float(distance)}
            for idx, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def get_documents(self, *, where: dict[str, Any] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        if limit is not None:
            kwargs["limit"] = limit
        result = self.collection.get(**kwargs)
        return [
            {"id": idx, "text": text, "metadata": metadata or {}}
            for idx, text, metadata in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    def delete_source(self, session_id: str, source_id: str) -> None:
        self.collection.delete(where={"$and": [{"session_id": session_id}, {"source_id": source_id}]})
