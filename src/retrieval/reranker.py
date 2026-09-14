from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict]
    ) -> list[dict]:

        pairs = [
            (query, result["text"])
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):

            reranked_results.append({
                **result,
                "reranker_score": float(score)
            })

        reranked_results.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return reranked_results