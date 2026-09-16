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


DEPTHS = [10, 20, 50]


def get_relevant_ranks(results, relevant_chunks):
    """
    Return the ranks of all relevant chunks.
    """

    relevant_chunks = set(relevant_chunks)

    ranks = []

    for rank, result in enumerate(results, start=1):

        if result["chunk_index"] in relevant_chunks:
            ranks.append(rank)

    return ranks


def main():

    # --------------------------------------------------
    # Load document
    # --------------------------------------------------

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
    # BM25
    # --------------------------------------------------

    bm25_retriever = BM25Retriever(chunks)

    # --------------------------------------------------
    # Hybrid
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

    print("\n" + "=" * 80)
    print("QUERY-LEVEL MINIMUM CANDIDATE DEPTH")
    print("=" * 80)

    all_results = []

    # --------------------------------------------------
    # Evaluate every question
    # --------------------------------------------------

    for item in questions:

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        print("\n" + "-" * 80)
        print(f"Question: {question}")
        print(f"Relevant chunks: {relevant_chunks}")
        print("-" * 80)

        question_results = {}

        minimum_depth = None

        for depth in DEPTHS:

            candidates = hybrid_retriever.retrieve(
                question,
                top_k=depth,
                candidate_k=depth
            )

            # Check whether relevant evidence
            # exists BEFORE reranking.
            candidate_indices = {
                result["chunk_index"]
                for result in candidates
            }

            candidate_found = bool(
                candidate_indices &
                set(relevant_chunks)
            )

            # Rerank
            reranked = reranker.rerank(
                question,
                candidates
            )

            relevant_ranks = get_relevant_ranks(
                reranked,
                relevant_chunks
            )

            top_1_found = bool(
                set(
                    result["chunk_index"]
                    for result in reranked[:1]
                )
                &
                set(relevant_chunks)
            )

            top_3_found = bool(
                set(
                    result["chunk_index"]
                    for result in reranked[:3]
                )
                &
                set(relevant_chunks)
            )

            top_5_found = bool(
                set(
                    result["chunk_index"]
                    for result in reranked[:5]
                )
                &
                set(relevant_chunks)
            )

            question_results[depth] = {
                "candidate_found": candidate_found,
                "relevant_ranks": relevant_ranks,
                "top_1": top_1_found,
                "top_3": top_3_found,
                "top_5": top_5_found
            }

            print(
                f"\nDepth Top-{depth}"
            )

            print(
                f"Candidate found: {candidate_found}"
            )

            print(
                f"Relevant ranks after reranking: "
                f"{relevant_ranks}"
            )

            print(
                f"Top-1: {top_1_found}"
            )

            print(
                f"Top-3: {top_3_found}"
            )

            print(
                f"Top-5: {top_5_found}"
            )

            # Minimum depth means:
            # relevant evidence is reachable
            # AND reranker puts it in Top-5.
            if (
                minimum_depth is None
                and candidate_found
                and top_5_found
            ):
                minimum_depth = depth

        all_results.append({
            "question": question,
            "relevant_chunks": relevant_chunks,
            "results": question_results,
            "minimum_depth": minimum_depth
        })

        print(
            f"\nMinimum depth needed for Top-5 evidence: "
            f"{minimum_depth}"
        )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("MINIMUM DEPTH SUMMARY")
    print("=" * 80)

    print(
        "\nQuestion | Top-10 | Top-20 | Top-50 | Minimum depth"
    )

    print("-" * 80)

    for item in all_results:

        question = item["question"]

        status = []

        for depth in DEPTHS:

            result = item["results"][depth]

            if result["top_5"]:
                status.append("YES")
            else:
                status.append("NO")

        print(
            f"\n{question}"
        )

        print(
            f"Top-10={status[0]} | "
            f"Top-20={status[1]} | "
            f"Top-50={status[2]} | "
            f"Minimum={item['minimum_depth']}"
        )

    # --------------------------------------------------
    # Count minimum depths
    # --------------------------------------------------

    counts = {
        10: 0,
        20: 0,
        50: 0,
        None: 0
    }

    for item in all_results:
        counts[item["minimum_depth"]] += 1

    print("\n" + "=" * 80)
    print("DEPTH DISTRIBUTION")
    print("=" * 80)

    print(
        f"Minimum depth 10: "
        f"{counts[10]}/{len(all_results)}"
    )

    print(
        f"Minimum depth 20: "
        f"{counts[20]}/{len(all_results)}"
    )

    print(
        f"Minimum depth 50: "
        f"{counts[50]}/{len(all_results)}"
    )

    print(
        f"Not solved: "
        f"{counts[None]}/{len(all_results)}"
    )


if __name__ == "__main__":
    main()