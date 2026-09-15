import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"


def get_first_relevant_rank(results, relevant_chunks):
    """
    Return the rank of the first relevant chunk.
    Return None if no relevant chunk exists in the results.
    """

    relevant_chunks = set(relevant_chunks)

    for rank, result in enumerate(results, start=1):

        if result["chunk_index"] in relevant_chunks:
            return rank

    return None


def main():

    # --------------------------------------------------
    # 1. Load document
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
    # 3. Embeddings
    # --------------------------------------------------

    embedder = Embedder()

    embeddings = embedder.embed(chunks)

    print(f"Embeddings: {embeddings.shape}")

    # --------------------------------------------------
    # 4. Vector store
    # --------------------------------------------------

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(embeddings)

    # --------------------------------------------------
    # 5. Retriever
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
    # 7. Load questions
    # --------------------------------------------------

    with open(QUESTIONS_PATH, "r") as file:
        questions = json.load(file)

    # --------------------------------------------------
    # 8. Counters
    # --------------------------------------------------

    improved = 0
    worsened = 0
    unchanged = 0
    not_retrieved = 0

    # --------------------------------------------------
    # 9. Analyze every question
    # --------------------------------------------------

    for item in questions:

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        # ----------------------------------------------
        # Dense Top-20
        # ----------------------------------------------

        dense_results = retriever.retrieve(
            question,
            top_k=20
        )

        # ----------------------------------------------
        # Reranked Top-20
        # ----------------------------------------------

        reranked_results = reranker.rerank(
            question,
            dense_results
        )

        # ----------------------------------------------
        # Relevant evidence ranks
        # ----------------------------------------------

        dense_rank = get_first_relevant_rank(
            dense_results,
            relevant_chunks
        )

        reranked_rank = get_first_relevant_rank(
            reranked_results,
            relevant_chunks
        )

        # ----------------------------------------------
        # Determine outcome
        # ----------------------------------------------

        if dense_rank is None:

            if reranked_rank is None:
                category = "RETRIEVAL FAILURE"
                not_retrieved += 1

            else:
                category = "UNEXPECTED RERANKER ENTRY"

        elif reranked_rank < dense_rank:

            category = "RERANKING IMPROVED"
            improved += 1

        elif reranked_rank > dense_rank:

            category = "RERANKING WORSENED"
            worsened += 1

        else:

            category = "UNCHANGED"
            unchanged += 1

        # ----------------------------------------------
        # Print analysis
        # ----------------------------------------------

        print("\n" + "=" * 75)

        print("QUESTION")
        print("=" * 75)

        print(question)

        print("\nRelevant chunks:")
        print(relevant_chunks)

        print("\nDense rank:")
        print(dense_rank)

        print("Reranked rank:")
        print(reranked_rank)

        print("\nDiagnosis:")
        print(category)

        # ----------------------------------------------
        # Show candidate positions
        # ----------------------------------------------

        dense_top20 = [
            result["chunk_index"]
            for result in dense_results
        ]

        print("\nDense Top-20:")
        print(dense_top20)

        print("\nReranked Top-10:")

        reranked_top10 = [
            result["chunk_index"]
            for result in reranked_results[:10]
        ]

        print(reranked_top10)

    # --------------------------------------------------
    # 10. Final failure report
    # --------------------------------------------------

    total = len(questions)

    print("\n\n")
    print("=" * 75)
    print("RERANKING FAILURE ANALYSIS")
    print("=" * 75)

    print(f"\nTotal questions: {total}")

    print(
        f"\nReranking improved: "
        f"{improved}/{total} "
        f"({improved / total:.1%})"
    )

    print(
        f"Reranking worsened: "
        f"{worsened}/{total} "
        f"({worsened / total:.1%})"
    )

    print(
        f"Unchanged: "
        f"{unchanged}/{total} "
        f"({unchanged / total:.1%})"
    )

    print(
        f"Retrieval failures: "
        f"{not_retrieved}/{total} "
        f"({not_retrieved / total:.1%})"
    )

    print("\nInterpretation:")

    print(
        "RERANKING IMPROVED = relevant evidence was retrieved "
        "but ranked higher after reranking."
    )

    print(
        "RERANKING WORSENED = relevant evidence moved lower "
        "after reranking."
    )

    print(
        "UNCHANGED = relevant evidence stayed at the same rank."
    )

    print(
        "RETRIEVAL FAILURE = relevant evidence was not present "
        "in the Dense Top-20 candidate set."
    )


if __name__ == "__main__":
    main()