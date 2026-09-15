class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        bm25_retriever
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60
    ):
        # Retrieve candidates from both systems
        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=candidate_k
        )

        bm25_results = self.bm25_retriever.retrieve(
            query,
            top_k=candidate_k
        )

        # Store RRF scores by chunk index
        rrf_scores = {}

        # Dense ranking contribution
        for rank, result in enumerate(dense_results, start=1):

            chunk_index = result["chunk_index"]

            rrf_scores.setdefault(
                chunk_index,
                0.0
            )

            rrf_scores[chunk_index] += (
                1 / (rrf_k + rank)
            )

        # BM25 ranking contribution
        for rank, result in enumerate(bm25_results, start=1):

            chunk_index = result["chunk_index"]

            rrf_scores.setdefault(
                chunk_index,
                0.0
            )

            rrf_scores[chunk_index] += (
                1 / (rrf_k + rank)
            )

        # Sort chunks by combined RRF score
        ranked_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for chunk_index, score in ranked_chunks[:top_k]:

            results.append({
                "chunk_index": int(chunk_index),
                "score": float(score),
                "text": self.dense_retriever.chunks[chunk_index]
            })

        return results