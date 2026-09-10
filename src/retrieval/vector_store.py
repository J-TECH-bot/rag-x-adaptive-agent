import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray):
        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ):
        query_embedding = query_embedding.astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        return scores, indices