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

    return int(bool(retrieved_chunks & relevant_chunks))


def reciprocal_rank(results, relevant_chunks):
    relevant_chunks = set(relevant_chunks)

    for rank, result in enumerate(results, start=1):
        if result["chunk_index"] in relevant_chunks:
            return 1 / rank

    return 0.0


def get_rank(results, relevant_chunks):
    relevant_chunks = set(relevant_chunks)

    for rank, result in enumerate(results, start=1):
        if result["chunk_index"] in relevant_chunks:
            return rank

    return None


def main():

    # --------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------

    text = load_pdf(PDF_PATH)

    # --------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Characters: {len(text)}")
    print(f"Chunks: {len(chunks)}")

    # --------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------

    embedder = Embedder()

    embeddings = embedder.embed(chunks)

    print(f"Embeddings: {embeddings.shape}")

    # --------------------------------------------------
    # 4. Create vector store
    # --------------------------------------------------

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(embeddings)

    # --------------------------------------------------
    # 5. Dense retriever
    # --------------------------------------------------

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks
    )

    # --------------------------------------------------
    # 6. Reranker
    # --------------------------------------------------

    reranker = Reranker()

    # --------------------------------------------------
    # 7. Load evaluation questions
    # --------------------------------------------------

    with open(QUESTIONS_PATH, "r") as file:
        questions = json.load(file)

    # --------------------------------------------------
    # 8. Metric storage
    # --------------------------------------------------

    dense_totals = {
        1: 0,
        3: 0,
        5: 0
    }

    reranked_totals = {
        1: 0,
        3: 0,
        5: 0
    }

    dense_mrr = 0.0
    reranked_mrr = 0.0

    # --------------------------------------------------
    # 9. Run experiment
    # --------------------------------------------------

    for item in questions:

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        # ----------------------------------------------
        # Dense retrieval
        # ----------------------------------------------

        dense_results = retriever.retrieve(
            question,
            top_k=20
        )

        # ----------------------------------------------
        # Reranking
        # ----------------------------------------------

        reranked_results = reranker.rerank(
            question,
            dense_results
        )

        # ----------------------------------------------
        # Find relevant chunk rank
        # ----------------------------------------------

        dense_rank = get_rank(
            dense_results,
            relevant_chunks
        )

        reranked_rank = get_rank(
            reranked_results,
            relevant_chunks
        )

        print("\n" + "=" * 70)
        print("QUESTION")
        print("=" * 70)

        print(question)

        print("\nRelevant chunks:")
        print(relevant_chunks)

        print("\nDense Top-10:")
        print([
            result["chunk_index"]
            for result in dense_results[:10]
        ])

        print("\nReranked Top-10:")
        print([
            result["chunk_index"]
            for result in reranked_results[:10]
        ])

        print("\nRelevant evidence rank:")

        print("Dense:", dense_rank)
        print("Reranked:", reranked_rank)

        # ----------------------------------------------
        # Rank movement
        # ----------------------------------------------

        if dense_rank is None:
            if reranked_rank is None:
                movement = "Not retrieved"
            else:
                movement = "Entered through reranking"
        elif reranked_rank is None:
            movement = "Lost after reranking"
        elif reranked_rank < dense_rank:
            movement = "Improved"
        elif reranked_rank > dense_rank:
            movement = "Worsened"
        else:
            movement = "Unchanged"

        print("Movement:", movement)

        # ----------------------------------------------
        # Recall metrics
        # ----------------------------------------------

        for k in [1, 3, 5]:

            dense_score = recall_at_k(
                dense_results,
                relevant_chunks,
                k
            )

            reranked_score = recall_at_k(
                reranked_results,
                relevant_chunks,
                k
            )

            dense_totals[k] += dense_score
            reranked_totals[k] += reranked_score

        # ----------------------------------------------
        # MRR
        # ----------------------------------------------

        dense_rr = reciprocal_rank(
            dense_results,
            relevant_chunks
        )

        reranked_rr = reciprocal_rank(
            reranked_results,
            relevant_chunks
        )

        dense_mrr += dense_rr
        reranked_mrr += reranked_rr

    # --------------------------------------------------
    # 10. Final comparison
    # --------------------------------------------------

    total_questions = len(questions)

    print("\n\n")
    print("=" * 70)
    print("RERANKING ABLATION RESULTS")
    print("=" * 70)

    print("\nMetric              Dense       Reranked       Change")
    print("-" * 70)

    for k in [1, 3, 5]:

        dense_recall = (
            dense_totals[k] / total_questions
        )

        reranked_recall = (
            reranked_totals[k] / total_questions
        )

        change = reranked_recall - dense_recall

        print(
            f"Recall@{k:<12} "
            f"{dense_recall:.3f}       "
            f"{reranked_recall:.3f}       "
            f"{change:+.3f}"
        )

    dense_mrr_avg = dense_mrr / total_questions
    reranked_mrr_avg = reranked_mrr / total_questions

    print(
        f"MRR{' ':<14} "
        f"{dense_mrr_avg:.3f}       "
        f"{reranked_mrr_avg:.3f}       "
        f"{reranked_mrr_avg - dense_mrr_avg:+.3f}"
    )


if __name__ == "__main__":
    main()