def recall_at_k(results, relevant_chunks, k):
    retrieved_chunks = {
        result["chunk_index"]
        for result in results[:k]
    }

    relevant_chunks = set(relevant_chunks)

    return int(bool(retrieved_chunks & relevant_chunks))


def reciprocal_rank(results, relevant_chunks):
    relevant_chunks = set(relevant_chunks)

    for rank, result in enumerate(results, start=1):
        if result["chunk_index"] in relevant_chunks:
            return 1 / rank

    return 0.0