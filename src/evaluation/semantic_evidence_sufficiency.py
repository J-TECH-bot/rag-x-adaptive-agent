import numpy as np


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two embedding vectors.
    """

    a = np.asarray(a)
    b = np.asarray(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def semantic_evidence_score(
    query: str,
    results: list[dict],
    embedder
) -> dict:
    """
    Calculate semantic similarity between the query
    and each retrieved evidence chunk.

    The final score is the maximum similarity among
    the retrieved chunks.
    """

    if not results:
        return {
            "score": 0.0,
            "chunk_scores": []
        }

    query_embedding = embedder.embed([query])[0]

    evidence_texts = [
        result["text"]
        for result in results
    ]

    evidence_embeddings = embedder.embed(evidence_texts)

    chunk_scores = []

    for result, embedding in zip(
        results,
        evidence_embeddings
    ):
        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        chunk_scores.append({
            "chunk_index": result["chunk_index"],
            "similarity": similarity
        })

    chunk_scores.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    max_score = chunk_scores[0]["similarity"]

    return {
        "score": float(max_score),
        "chunk_scores": chunk_scores
    }