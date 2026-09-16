import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"


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


def evaluate_depth(
    depth,
    hybrid_retriever,
    reranker,
    questions
):
    recall_1 = []
    recall_3 = []
    recall_5 = []
    mrr_scores = []

    for item in questions:

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        # Hybrid candidate retrieval
        candidates = hybrid_retriever.retrieve(
            question,
            top_k=depth,
            candidate_k=depth
        )

        # Cross-encoder reranking
        reranked = reranker.rerank(
            question,
            candidates
        )

        recall_1.append(
            recall_at_k(
                reranked,
                relevant_chunks,
                1
            )
        )

        recall_3.append(
            recall_at_k(
                reranked,
                relevant_chunks,
                3
            )
        )

        recall_5.append(
            recall_at_k(
                reranked,
                relevant_chunks,
                5
            )
        )

        mrr_scores.append(
            reciprocal_rank(
                reranked,
                relevant_chunks
            )
        )

        relevant_ranks = []

        for rank, result in enumerate(
            reranked,
            start=1
        ):
            if result["chunk_index"] in relevant_chunks:
                relevant_ranks.append(rank)

        print(
            f"\nQuestion: {question}"
        )

        print(
            f"Relevant chunks: {relevant_chunks}"
        )

        print(
            f"Relevant ranks after reranking: "
            f"{relevant_ranks}"
        )

        print(
            f"Top-5 after reranking: "
            f"{[r['chunk_index'] for r in reranked[:5]]}"
        )

    return {
        "recall@1": sum(recall_1) / len(recall_1),
        "recall@3": sum(recall_3) / len(recall_3),
        "recall@5": sum(recall_5) / len(recall_5),
        "mrr": sum(mrr_scores) / len(mrr_scores)
    }


def main():

    text = load_pdf(PDF_PATH)

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Total chunks: {len(chunks)}")

    # --------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------

    embedder = Embedder()

    document_embeddings = embedder.embed(chunks)

    vector_store = VectorStore(
        dimension=document_embeddings.shape[1]
    )

    vector_store.add(document_embeddings)

    dense_retriever = Retriever(
        embedder,
        vector_store,
        chunks
    )

    # --------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------

    bm25_retriever = BM25Retriever(chunks)

    # --------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever
    )

    # --------------------------------------------------
    # Reranker
    # --------------------------------------------------

    reranker = Reranker()

    # --------------------------------------------------
    # Questions
    # --------------------------------------------------

    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    depths = [10, 20, 50]

    summary = {}

    print("\n" + "=" * 75)
    print("HYBRID CANDIDATE DEPTH × RERANKING")
    print("=" * 75)

    for depth in depths:

        print("\n")
        print("-" * 75)
        print(f"HYBRID TOP-{depth} → RERANKER")
        print("-" * 75)

        metrics = evaluate_depth(
            depth,
            hybrid_retriever,
            reranker,
            questions
        )

        summary[depth] = metrics

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    print(
        "\nDepth       Recall@1   Recall@3   Recall@5   MRR"
    )

    print("-" * 60)

    for depth in depths:

        metrics = summary[depth]

        print(
            f"Top-{depth:<6} "
            f"{metrics['recall@1']:.3f}       "
            f"{metrics['recall@3']:.3f}       "
            f"{metrics['recall@5']:.3f}       "
            f"{metrics['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()