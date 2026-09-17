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
    """

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


# ============================================================
# QUERY DIFFICULTY ANALYSIS
# ============================================================

all_results = []


for question_number, question in enumerate(questions, start=1):

    query = question["question"]
    relevant_chunks = question.get("relevant_chunks", [])

    print()
    print("=" * 70)
    print(f"Question {question_number}/{len(questions)}")
    print(f"Query: {query}")
    print(f"Relevant chunks: {relevant_chunks}")
    print("=" * 70)

    question_result = {
        "question_id": question.get(
            "id",
            question_number
        ),
        "question": query,
        "relevant_chunks": relevant_chunks,
        "depth_analysis": {}
    }

    # --------------------------------------------------------
    # RETRIEVAL DEPTH ANALYSIS
    # --------------------------------------------------------

    for depth in DEPTHS:

        print(f"  Analyzing retrieval depth: {depth}")

        dense_results = get_dense_results(
            query,
            depth
        )

        bm25_results = get_bm25_results(
            query,
            depth
        )

        hybrid_results = get_hybrid_results(
            query,
            depth
        )

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

        # ----------------------------------------------------
        # RERANKING
        # ----------------------------------------------------

        reranked_results = rerank_results(
            query,
            hybrid_results
        )

        depth_result["reranked"] = {
            "contains_relevant": contains_relevant(
                reranked_results,
                relevant_chunks
            ),
            "relevant_rank": get_relevant_rank(
                reranked_results,
                relevant_chunks
            )
        }

        question_result["depth_analysis"][str(depth)] = depth_result

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
    # QUERY-LEVEL DIFFICULTY SIGNALS
    # ========================================================

    depth_10 = question_result["depth_analysis"]["10"]
    depth_20 = question_result["depth_analysis"]["20"]
    depth_50 = question_result["depth_analysis"]["50"]

    hybrid_at_10 = depth_10["hybrid"]["contains_relevant"]
    hybrid_at_20 = depth_20["hybrid"]["contains_relevant"]
    hybrid_at_50 = depth_50["hybrid"]["contains_relevant"]

    question_result["difficulty_signals"] = {

        # Evidence cannot be found even after searching
        # a large candidate set.
        "unreachable_at_50": not hybrid_at_50,

        # Evidence is not in the first 10 but appears
        # when retrieval depth increases.
        "depth_sensitive": (
            not hybrid_at_10
            and hybrid_at_50
        ),

        # Evidence is already reachable at Top-10.
        "easy_at_top10": hybrid_at_10,

        # Reranking changed the position of evidence.
        "reranking_helped": (
            depth_10["hybrid"]["relevant_rank"] is not None
            and depth_10["reranked"]["relevant_rank"] is not None
            and depth_10["reranked"]["relevant_rank"]
            < depth_10["hybrid"]["relevant_rank"]
        ),

        # Dense and BM25 retrieval disagree substantially.
        "low_dense_bm25_agreement": (
            depth_10["dense_bm25_agreement"] < 0.3
        )
    }

    all_results.append(question_result)


# ============================================================
# SAVE RESULTS
# ============================================================

output = {
    "configuration": {
        "pdf_path": PDF_PATH,
        "questions_path": QUESTIONS_PATH,
        "depths": DEPTHS,
        "total_chunks": len(chunks),
        "total_questions": len(questions)
    },
    "questions": all_results
}


with open(OUTPUT_PATH, "w") as f:
    json.dump(
        output,
        f,
        indent=2
    )


print()
print("=" * 70)
print("QUERY DIFFICULTY ANALYSIS COMPLETE")
print("=" * 70)
print(f"Results saved to: {OUTPUT_PATH}")
print(f"Questions analyzed: {len(all_results)}")