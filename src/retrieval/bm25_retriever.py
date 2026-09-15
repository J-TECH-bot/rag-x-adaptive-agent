from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks: list[str]):
        self.chunks = chunks

        # Tokenize each chunk
        self.tokenized_chunks = [
            chunk.lower().split()
            for chunk in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):
        # Tokenize query
        tokenized_query = query.lower().split()

        # Calculate BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices sorted by score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "chunk_index": index,
                "score": float(scores[index]),
                "text": self.chunks[index]
            })

        return results