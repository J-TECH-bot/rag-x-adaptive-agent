import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"


def recall_at_k(results, relevant_chunks, k):

    retrieved_chunks = {
        result["chunk_index"]
        for result in results[:k]
    }

    relevant_chunks = set(relevant_chunks)

    return int(
        bool(retrieved_chunks & relevant_chunks)
    )


def reciprocal_rank(results, relevant_chunks):

    relevant_chunks = set(relevant_chunks)

    for rank, result in enumerate(results, start=1):

        if result["chunk_index"] in relevant_chunks:
            return 1 / rank

    return 0.0


def main():

    # -----------------------------
    # 1. Load PDF
    # -----------------------------
    text = load_pdf(PDF_PATH)

    # -----------------------------
    # 2. Create chunks
    # -----------------------------
    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Characters: {len(text)}")
    print(f"Chunks: {len(chunks)}")

    # -----------------------------
    # 3. Create embeddings
    # -----------------------------
    embedder = Embedder()

    embeddings = embedder.embed(chunks)

    print(f"Embeddings: {embeddings.shape}")

    # -----------------------------
    # 4. Create vector store
    # -----------------------------
    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(embeddings)

    print(f"Vectors stored: {len(chunks)}")

    # -----------------------------
    # 5. Create dense retriever
    # -----------------------------
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks
    )

    # -----------------------------
    # 6. Create reranker
    # -----------------------------
    reranker = Reranker()

    # -----------------------------
    # 7. Load questions
    # -----------------------------
    with open(QUESTIONS_PATH, "r") as file:
        questions = json.load(file)

    recall_totals = {
        1: 0,
        3: 0,
        5: 0
    }

    mrr_total = 0.0

    # -----------------------------
    # 8. Evaluate each question
    # -----------------------------
    for item in questions:

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        # Dense retrieval
        dense_results = retriever.retrieve(
            question,
            top_k=20
        )

        # Reranking
        reranked_results = reranker.rerank(
            question,
            dense_results
        )

        print("\nQuestion:")
        print(question)

        print(
            "Relevant chunks:",
            relevant_chunks
        )

        print(
            "Dense Top-5:",
            [
                result["chunk_index"]
                for result in dense_results[:5]
            ]
        )

        print(
            "Reranked Top-5:",
            [
                result["chunk_index"]
                for result in reranked_results[:5]
            ]
        )

        # Recall@K
        for k in recall_totals:

            score = recall_at_k(
                reranked_results,
                relevant_chunks,
                k
            )

            recall_totals[k] += score

            print(
                f"Reranked Recall@{k}: {score}"
            )

        # MRR
        rr = reciprocal_rank(
            reranked_results,
            relevant_chunks
        )

        mrr_total += rr

        print(
            f"Reranked MRR: {rr:.3f}"
        )

    # -----------------------------
    # 9. Final results
    # -----------------------------
    total_questions = len(questions)

    print("\n")
    print("=" * 60)
    print("RERANKING EVALUATION")
    print("=" * 60)

    for k in recall_totals:

        recall = (
            recall_totals[k]
            / total_questions
        )

        print(
            f"Recall@{k}: {recall:.3f}"
        )

    mrr = mrr_total / total_questions

    print(
        f"MRR:      {mrr:.3f}"
    )


if __name__ == "__main__":
    main()