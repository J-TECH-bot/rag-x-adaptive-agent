from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunks: list[str]
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunks = chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):
        query_embedding = self.embedder.embed([query])

        scores, indices = self.vector_store.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            results.append({
                "chunk_index": int(index),
                "score": float(score),
                "text": self.chunks[index]
            })

        return results