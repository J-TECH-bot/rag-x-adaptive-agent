
import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text

from embeddings.embedder import Embedder

from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"
OUTPUT_PATH = "data/evaluation/query_difficulty_analysis.json"

DEPTHS = [10, 20, 50]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading PDF...")

text = load_pdf(PDF_PATH)

chunks = chunk_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Total chunks: {len(chunks)}")


with open(QUESTIONS_PATH, "r") as f:
    questions = json.load(f)

print(f"Total evaluation questions: {len(questions)}")


# ============================================================
# BUILD EMBEDDINGS + RETRIEVERS
# ============================================================

print("Building embeddings...")

embedder = Embedder()

document_embeddings = embedder.embed(chunks)

vector_store = VectorStore(
    dimension=document_embeddings.shape[1]
)

vector_store.add(document_embeddings)


dense_retriever = Retriever(
    embedder=embedder,
    vector_store=vector_store,
    chunks=chunks
)


bm25_retriever = BM25Retriever(chunks)


hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    bm25_retriever=bm25_retriever
)


print("Loading reranker...")

reranker = Reranker()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_dense_results(query, depth):
    return dense_retriever.retrieve(
        query,
        top_k=depth
    )


def get_bm25_results(query, depth):
    return bm25_retriever.retrieve(
        query,
        top_k=depth
    )


def get_hybrid_results(query, depth):
    return hybrid_retriever.retrieve(
        query,
        top_k=depth,
        candidate_k=depth
    )


def get_score_gap(results):
    """
    Difference between the first and second retrieval scores.
    """

    if len(results) < 2:
        return 0.0

    return results[0]["score"] - results[1]["score"]


def contains_relevant(results, relevant_chunks):
    """
    Check whether at least one relevant chunk
    appears in the retrieved results.

    For unanswerable questions, relevant_chunks is empty,
    so this returns False.
    """

    if not relevant_chunks:
        return False

    retrieved = {
        result["chunk_index"]
        for result in results
    }

    relevant = set(relevant_chunks)

    return bool(retrieved & relevant)


def get_relevant_rank(results, relevant_chunks):
    """
    Return the rank of the first relevant chunk.

    Returns None if no relevant chunk is found.
    """

    if not relevant_chunks:
        return None

    relevant = set(relevant_chunks)

    for rank, result in enumerate(results, start=1):

        if result["chunk_index"] in relevant:
            return rank

    return None


def calculate_agreement(dense_results, bm25_results):
    """
    Calculate Jaccard similarity between
    Dense Top-10 and BM25 Top-10.
    """

    dense_top10 = {
        result["chunk_index"]
        for result in dense_results[:10]
    }

    bm25_top10 = {
        result["chunk_index"]
        for result in bm25_results[:10]
    }

    intersection = dense_top10 & bm25_top10
    union = dense_top10 | bm25_top10

    if not union:
        return 0.0

    return len(intersection) / len(union)


def rerank_results(query, results):
    """
    Rerank retrieved results using the cross-encoder reranker.
    """

    if not results:
        return []

    return reranker.rerank(
        query,
        results
    )


def get_reranking_improvement(original_rank, reranked_rank):
    """
    Calculate how many positions the relevant evidence moved.

    Positive value = moved upward.
    Zero = no change.
    Negative value = moved downward.
    """

    if original_rank is None or reranked_rank is None:
        return None

    return original_rank - reranked_rank


# ============================================================
# QUERY DIFFICULTY ANALYSIS
# ============================================================

all_results = []


for question_number, question in enumerate(questions, start=1):

    query = question["question"]

    answerability = question.get(
        "answerability",
        "unknown"
    )

    difficulty_type = question.get(
        "difficulty_type",
        "unknown"
    )

    relevant_chunks = question.get(
        "relevant_chunks",
        []
    )

    reference_answer = question.get(
        "reference_answer",
        ""
    )

    print()
    print("=" * 70)
    print(f"Question {question_number}/{len(questions)}")
    print(f"Query: {query}")
    print(f"Answerability: {answerability}")
    print(f"Difficulty type: {difficulty_type}")
    print(f"Relevant chunks: {relevant_chunks}")
    print("=" * 70)

    question_result = {
        "question_id": question.get(
            "id",
            question_number
        ),

        "question": query,

        "answerability": answerability,

        "difficulty_type": difficulty_type,

        "relevant_chunks": relevant_chunks,

        "reference_answer": reference_answer,

        "depth_analysis": {}
    }


    # --------------------------------------------------------
    # RETRIEVAL DEPTH ANALYSIS
    # --------------------------------------------------------

    for depth in DEPTHS:

        print(
            f"  Analyzing retrieval depth: {depth}"
        )


        # ====================================================
        # DENSE RETRIEVAL
        # ====================================================

        dense_results = get_dense_results(
            query,
            depth
        )


        # ====================================================
        # BM25 RETRIEVAL
        # ====================================================

        bm25_results = get_bm25_results(
            query,
            depth
        )


        # ====================================================
        # HYBRID RETRIEVAL
        # ====================================================

        hybrid_results = get_hybrid_results(
            query,
            depth
        )


        # ====================================================
        # RERANK HYBRID RESULTS
        # ====================================================

        reranked_results = rerank_results(
            query,
            hybrid_results
        )


        # ====================================================
        # STORE RETRIEVAL METRICS
        # ====================================================

        depth_result = {

            "dense": {

                "contains_relevant": contains_relevant(
                    dense_results,
                    relevant_chunks
                ),

                "relevant_rank": get_relevant_rank(
                    dense_results,
                    relevant_chunks
                ),

                "score_gap": get_score_gap(
                    dense_results
                )
            },


            "bm25": {

                "contains_relevant": contains_relevant(
                    bm25_results,
                    relevant_chunks
                ),

                "relevant_rank": get_relevant_rank(
                    bm25_results,
                    relevant_chunks
                ),

                "score_gap": get_score_gap(
                    bm25_results
                )
            },


            "hybrid": {

                "contains_relevant": contains_relevant(
                    hybrid_results,
                    relevant_chunks
                ),

                "relevant_rank": get_relevant_rank(
                    hybrid_results,
                    relevant_chunks
                ),

                "score_gap": get_score_gap(
                    hybrid_results
                )
            },


            "dense_bm25_agreement": calculate_agreement(
                dense_results,
                bm25_results
            )
        }


        # ====================================================
        # RERANKING METRICS
        # ====================================================

        original_rank = get_relevant_rank(
            hybrid_results,
            relevant_chunks
        )

        reranked_rank = get_relevant_rank(
            reranked_results,
            relevant_chunks
        )


        depth_result["reranked"] = {

            "contains_relevant": contains_relevant(
                reranked_results,
                relevant_chunks
            ),

            "relevant_rank": reranked_rank,

            "rank_improvement": get_reranking_improvement(
                original_rank,
                reranked_rank
            )
        }


        # ====================================================
        # SAVE DEPTH RESULT
        # ====================================================

        question_result["depth_analysis"][
            str(depth)
        ] = depth_result


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"    Dense relevant rank: "
            f"{depth_result['dense']['relevant_rank']}"
        )

        print(
            f"    BM25 relevant rank: "
            f"{depth_result['bm25']['relevant_rank']}"
        )

        print(
            f"    Hybrid relevant rank: "
            f"{depth_result['hybrid']['relevant_rank']}"
        )

        print(
            f"    Reranked relevant rank: "
            f"{depth_result['reranked']['relevant_rank']}"
        )

        print(
            f"    Dense/BM25 agreement: "
            f"{depth_result['dense_bm25_agreement']:.3f}"
        )


    # ========================================================
    # GET DEPTH RESULTS
    # ========================================================

    depth_10 = question_result[
        "depth_analysis"
    ]["10"]

    depth_20 = question_result[
        "depth_analysis"
    ]["20"]

    depth_50 = question_result[
        "depth_analysis"
    ]["50"]


    # ========================================================
    # HYBRID EVIDENCE REACHABILITY
    # ========================================================

    hybrid_at_10 = depth_10[
        "hybrid"
    ]["contains_relevant"]

    hybrid_at_20 = depth_20[
        "hybrid"
    ]["contains_relevant"]

    hybrid_at_50 = depth_50[
        "hybrid"
    ]["contains_relevant"]


    # ========================================================
    # ANSWERABLE QUERY ANALYSIS
    # ========================================================

    if answerability == "answerable":

        retrieval_failure = not hybrid_at_50

        depth_sensitive = (
            not hybrid_at_10
            and hybrid_at_50
        )

        easy_at_top10 = hybrid_at_10


        # Evidence is retrieved in Top-50,
        # but not ranked in Top-10.
        ranking_failure = (
            hybrid_at_50
            and not hybrid_at_10
        )


        # Reranking improved the position
        # of relevant evidence at Top-10.
        reranking_helped = (
            depth_10["hybrid"]["relevant_rank"]
            is not None
            and depth_10["reranked"]["relevant_rank"]
            is not None
            and depth_10["reranked"]["relevant_rank"]
            < depth_10["hybrid"]["relevant_rank"]
        )


    # ========================================================
    # UNANSWERABLE QUERY ANALYSIS
    # ========================================================

    elif answerability == "unanswerable":

        retrieval_failure = None

        depth_sensitive = None

        easy_at_top10 = None

        ranking_failure = None

        reranking_helped = None


    # ========================================================
    # UNKNOWN ANSWERABILITY
    # ========================================================

    else:

        retrieval_failure = None

        depth_sensitive = None

        easy_at_top10 = None

        ranking_failure = None

        reranking_helped = None


    # ========================================================
    # QUERY-LEVEL DIFFICULTY SIGNALS
    # ========================================================

    question_result["difficulty_signals"] = {

        # ----------------------------------------------------
        # Relevant evidence cannot be found by Hybrid
        # even at Top-50.
        # ----------------------------------------------------

        "retrieval_failure_at_50": retrieval_failure,


        # ----------------------------------------------------
        # Evidence is absent at Top-10 but appears by Top-50.
        # ----------------------------------------------------

        "depth_sensitive": depth_sensitive,


        # ----------------------------------------------------
        # Evidence is already available at Top-10.
        # ----------------------------------------------------

        "easy_at_top10": easy_at_top10,


        # ----------------------------------------------------
        # Evidence is reachable in the larger candidate set
        # but is not ranked in the Top-10.
        # ----------------------------------------------------

        "ranking_failure": ranking_failure,


        # ----------------------------------------------------
        # Cross-encoder reranking improves evidence position.
        # ----------------------------------------------------

        "reranking_helped": reranking_helped,


        # ----------------------------------------------------
        # Dense and BM25 retrieve substantially different
        # candidates.
        # ----------------------------------------------------

        "low_dense_bm25_agreement": (
            depth_10[
                "dense_bm25_agreement"
            ] < 0.3
        )
    }


    # ========================================================
    # EVIDENCE REACHABILITY SUMMARY
    # ========================================================

    question_result["evidence_reachability"] = {

        "hybrid_top10": hybrid_at_10,

        "hybrid_top20": hybrid_at_20,

        "hybrid_top50": hybrid_at_50,

        "first_relevant_rank": (
            depth_50[
                "hybrid"
            ]["relevant_rank"]
        )
    }


    # ========================================================
    # UNANSWERABLE QUERY INFORMATION
    # ========================================================

    if answerability == "unanswerable":

        question_result[
            "unanswerable_analysis"
        ] = {

            "expected_answer": None,

            "relevant_chunks_expected": False,

            # This is intentionally NOT called
            # "safe_to_abstain".
            #
            # Evidence sufficiency will be implemented
            # in the next stage.
            "requires_evidence_sufficiency_check": True
        }


    # ========================================================
    # APPEND QUESTION RESULT
    # ========================================================

    all_results.append(
        question_result
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

answerable_questions = [
    result
    for result in all_results
    if result["answerability"] == "answerable"
]

unanswerable_questions = [
    result
    for result in all_results
    if result["answerability"] == "unanswerable"
]


retrieval_failures = sum(
    result["difficulty_signals"][
        "retrieval_failure_at_50"
    ]
    is True
    for result in answerable_questions
)


depth_sensitive_queries = sum(
    result["difficulty_signals"][
        "depth_sensitive"
    ]
    is True
    for result in answerable_questions
)


ranking_failures = sum(
    result["difficulty_signals"][
        "ranking_failure"
    ]
    is True
    for result in answerable_questions
)


reranking_helped_queries = sum(
    result["difficulty_signals"][
        "reranking_helped"
    ]
    is True
    for result in answerable_questions
)


easy_top10_queries = sum(
    result["difficulty_signals"][
        "easy_at_top10"
    ]
    is True
    for result in answerable_questions
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = {

    "configuration": {

        "pdf_path": PDF_PATH,

        "questions_path": QUESTIONS_PATH,

        "depths": DEPTHS,

        "total_chunks": len(chunks),

        "total_questions": len(questions),

        "answerable_questions": len(
            answerable_questions
        ),

        "unanswerable_questions": len(
            unanswerable_questions
        )
    },


    "summary": {

        "answerable_questions": len(
            answerable_questions
        ),

        "unanswerable_questions": len(
            unanswerable_questions
        ),

        "retrieval_failures_at_50": (
            retrieval_failures
        ),

        "depth_sensitive_queries": (
            depth_sensitive_queries
        ),

        "ranking_failures": (
            ranking_failures
        ),

        "reranking_helped_queries": (
            reranking_helped_queries
        ),

        "easy_top10_queries": (
            easy_top10_queries
        )
    },


    "questions": all_results
}


with open(
    OUTPUT_PATH,
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 70)

print(
    "QUERY DIFFICULTY ANALYSIS COMPLETE"
)

print("=" * 70)

print(
    f"Results saved to: {OUTPUT_PATH}"
)

print(
    f"Questions analyzed: {len(all_results)}"
)

print(
    f"Answerable: {len(answerable_questions)}"
)

print(
    f"Unanswerable: {len(unanswerable_questions)}"
)

print(
    f"Retrieval failures at Top-50: "
    f"{retrieval_failures}"
)

print(
    f"Depth-sensitive queries: "
    f"{depth_sensitive_queries}"
)

print(
    f"Ranking failures: "
    f"{ranking_failures}"
)

print(
    f"Reranking helped: "
    f"{reranking_helped_queries}"
)

print(
    f"Easy at Top-10: "
    f"{easy_top10_queries}"
)