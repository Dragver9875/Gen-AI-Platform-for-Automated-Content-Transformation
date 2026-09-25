import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from utils import create_chroma_client


# ==============================
# Embeddings + Chroma
# ==============================


class HybridRetriever:
    def __init__(self, chunks, chroma_dir: str = "./chroma_db", collection_name: str = "document_chunks"):
        self.chunks = chunks
        self.texts = [chunk["text"] for chunk in chunks]
        self.chunks_metadata = [chunk.get("metadata", {}) for chunk in chunks]

        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("microsoft/harrier-oss-v1-0.6b", model_kwargs={"dtype": "auto"})

        print("Loading reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.client = create_chroma_client(chroma_dir)
        # use an explicit, stable collection name to avoid unexpected internal ids
        self.collection_name = collection_name
        self.chroma_dir = chroma_dir

        self.build_chroma()
        self.build_bm25()

    def _get_collection(self):
        """Fetch the collection fresh from the client to avoid stale references."""
        return self.client.get_or_create_collection(name=self.collection_name)

    def build_chroma(self):
        embeddings = self.embedding_model.encode(self.texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

        ids = []
        seen = set()
        for i, chunk in enumerate(self.chunks):
            chunk_id = chunk.get("metadata", {}).get("chunk_id", i)
            id_str = str(chunk_id)
            if id_str in seen:
                id_str = f"{id_str}_{i}"
            ids.append(id_str)
            seen.add(id_str)
        self.ids = ids

        # If the collection already has documents, delete and recreate it.
        # Use the stable collection name when deleting to avoid errors where
        # the underlying collection object contains an internal UUID name.
        try:
            collection = self._get_collection()
            if collection.count() > 0:
                try:
                    self.client.delete_collection(name=self.collection_name)
                except Exception:
                    # If deletion fails because the collection does not exist, ignore
                    pass
                collection = self._get_collection()
        except Exception:
            # If counting fails (race or corrupted state), attempt to recreate collection
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            collection = self._get_collection()

        collection.add(
            ids=ids,
            documents=self.texts,
            metadatas=self.chunks_metadata,
            embeddings=embeddings,
        )

        print(f"Chroma built with {len(self.texts)} chunks")

    def build_bm25(self):
        tokenized_docs = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        print("BM25 built")

    def vector_search(self, query, top_k=10):
        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).tolist()

        collection = self._get_collection()
        search_result = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        ids = search_result["ids"][0]
        documents = search_result["documents"][0]
        metadatas = search_result["metadatas"][0]
        distances = search_result["distances"][0]

        for idx, text, metadata, score in zip(ids, documents, metadatas, distances):
            results.append(
                {
                    "id": idx,
                    "score": float(score),
                    "text": text,
                    "metadata": metadata,
                }
            )

        return results

    def bm25_search(self, query, top_k=10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []

        for idx in top_indices:
            results.append(
                {
                    "id": self.ids[idx],
                    "score": float(scores[idx]),
                    "text": self.texts[idx],
                    "metadata": self.chunks_metadata[idx],
                }
            )
        return results

    def hybrid_search(self, query, bm25_k=10, vector_k=10):
        bm25_results = self.bm25_search(query, bm25_k)
        vector_results = self.vector_search(query, vector_k)
        combined = {}

        for result in bm25_results + vector_results:
            chunk_id = result["id"]
            if chunk_id not in combined:
                combined[chunk_id] = result

        return list(combined.values())

    def rerank(self, query, retrieved_docs, top_k=5):
        pairs = [[query, doc["text"]] for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(retrieved_docs, scores):
            doc["rerank_score"] = float(score)

        retrieved_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return retrieved_docs[:top_k]

    def retrieve(self, query, final_k=5):
        candidates = self.hybrid_search(query, bm25_k=15, vector_k=15)
        final_docs = self.rerank(query, candidates, final_k)
        return final_docs
    

