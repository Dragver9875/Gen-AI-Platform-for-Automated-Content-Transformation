from __future__ import annotations

from retrieval.fusion import reciprocal_rank_fusion
from retrieval.hybrid_retrieval import HybridRetriever


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(i + 1), 0.0] for i, _ in enumerate(texts)]

    def embed_query(self, query):
        return [1.0, 0.0]


class FakeStore:
    def __init__(self):
        self.docs = []

    def upsert(self, ids, documents, metadatas, embeddings):
        self.docs = [
            {"id": idx, "text": doc, "metadata": meta, "embedding": emb}
            for idx, doc, meta, emb in zip(ids, documents, metadatas, embeddings)
        ]

    def vector_query(self, query_embedding, *, top_k, where=None):
        docs = self.get_documents(where=where)
        return [{**doc, "distance": float(i)} for i, doc in enumerate(docs[:top_k])]

    def get_documents(self, *, where=None, limit=None):
        docs = self.docs
        if where and "$and" in where:
            for clause in where["$and"]:
                key, value = next(iter(clause.items()))
                docs = [d for d in docs if d["metadata"].get(key) == value]
        return docs[:limit] if limit else docs


def test_rrf_combines_ranked_lists():
    a = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
    b = [{"id": "b", "text": "B"}, {"id": "c", "text": "C"}]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0]["id"] == "b"


def test_hybrid_retriever_indexes_and_filters():
    store = FakeStore()
    retriever = HybridRetriever(store, FakeEmbedder())
    chunks = [
        {"text": "ransomware mitigation guidance", "metadata": {"chunk_id": "a", "session_id": "s1", "source_id": "x"}},
        {"text": "flood response advisory", "metadata": {"chunk_id": "b", "session_id": "s2", "source_id": "y"}},
    ]
    assert retriever.index_chunks(chunks) == 2
    where = {"$and": [{"session_id": "s1"}, {"source_id": "x"}]}
    result = retriever.retrieve("ransomware", where=where, final_k=2)
    assert len(result) == 1
    assert result[0]["id"] == "a"


def test_phase12_facade_retrieval_filter_shape():
    from app.phase12 import Phase12Pipeline

    class DummyRouter:
        pass

    class DummyChunker:
        pass

    class CaptureRetriever:
        def retrieve(self, query, *, where, final_k):
            return [{"query": query, "where": where, "final_k": final_k}]

    pipeline = Phase12Pipeline(DummyRouter(), DummyChunker(), CaptureRetriever())
    result = pipeline.retrieve("hello", session_id="s1", source_id="src", final_k=3)
    assert result[0]["where"] == {"$and": [{"session_id": "s1"}, {"source_id": "src"}]}
