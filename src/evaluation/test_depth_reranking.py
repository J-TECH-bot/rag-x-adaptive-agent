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
    # 5. Create retriever
    # --------------------------------------------------

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks
    )

    # --------------------------------------------------
    # 6. Create reranker
    # --------------------------------------------------

    reranker = Reranker()

    # --------------------------------------------------
    # 7. Load questions
    # --------------------------------------------------

    with open(QUESTIONS_PATH, "r") as file:
        questions = json.load(file)

    # --------------------------------------------------
    # 8. Candidate depths
    # --------------------------------------------------

    candidate_depths = [10, 20, 50]

    results = {}

    # --------------------------------------------------
    # 9. Run each candidate depth
    # --------------------------------------------------

    for depth in candidate_depths:

        print("\n")
        print("=" * 70)
        print(f"DENSE TOP-{depth} → RERANKER")
        print("=" * 70)

        recall_totals = {
            1: 0,
            3: 0,
            5: 0
        }

        mrr_total = 0.0

        # ----------------------------------------------
        # Evaluate every question
        # ----------------------------------------------

        for item in questions:

            question = item["question"]
            relevant_chunks = item["relevant_chunks"]

            # Dense retrieval
            dense_results = retriever.retrieve(
                question,
                top_k=depth
            )

            # Reranking
            reranked_results = reranker.rerank(
                question,
                dense_results
            )

            print("\nQuestion:")
            print(question)

            print("Relevant chunks:")
            print(relevant_chunks)

            print("Reranked Top-5:")
            print([
                result["chunk_index"]
                for result in reranked_results[:5]
            ])

            # ------------------------------------------
            # Recall
            # ------------------------------------------

            for k in [1, 3, 5]:

                score = recall_at_k(
                    reranked_results,
                    relevant_chunks,
                    k
                )

                recall_totals[k] += score

            # ------------------------------------------
            # MRR
            # ------------------------------------------

            rr = reciprocal_rank(
                reranked_results,
                relevant_chunks
            )

            mrr_total += rr

        # ----------------------------------------------
        # Calculate averages
        # ----------------------------------------------

        total_questions = len(questions)

        recall_scores = {}

        for k in [1, 3, 5]:

            recall_scores[k] = (
                recall_totals[k]
                / total_questions
            )

        mrr = mrr_total / total_questions

        results[depth] = {
            "recall@1": recall_scores[1],
            "recall@3": recall_scores[3],
            "recall@5": recall_scores[5],
            "mrr": mrr
        }

    # --------------------------------------------------
    # 10. Final comparison
    # --------------------------------------------------

    print("\n\n")
    print("=" * 70)
    print("CANDIDATE DEPTH × RERANKING")
    print("=" * 70)

    print(
        "\nDepth       Recall@1   Recall@3   Recall@5   MRR"
    )

    print("-" * 70)

    for depth in candidate_depths:

        metrics = results[depth]

        print(
            f"Top-{depth:<6} "
            f"{metrics['recall@1']:.3f}       "
            f"{metrics['recall@3']:.3f}       "
            f"{metrics['recall@5']:.3f}       "
            f"{metrics['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()