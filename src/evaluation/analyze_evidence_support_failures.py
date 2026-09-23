
import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text

from embeddings.embedder import Embedder

from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

from evaluation.evidence_support import EvidenceSupportJudge


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_PATH = (
    "data/evaluation/retrieval_questions.json"
)

PDF_PATH = (
    "data/raw/rag_original.pdf"
)

OUTPUT_PATH = (
    "data/evaluation/evidence_support_failure_analysis.json"
)

TOP_K = 10
CANDIDATE_K = 50

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# LOAD EVALUATION QUESTIONS
# ============================================================

with open(QUESTIONS_PATH, "r") as f:
    questions = json.load(f)


# ============================================================
# LOAD AND CHUNK DOCUMENT
# ============================================================

text = load_pdf(PDF_PATH)

chunks = chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

print(f"Chunks loaded: {len(chunks)}")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

embedder = Embedder()

chunk_embeddings = embedder.embed(chunks)


# ============================================================
# BUILD VECTOR STORE
# ============================================================

vector_store = VectorStore(
    dimension=chunk_embeddings.shape[1]
)

vector_store.add(chunk_embeddings)


# ============================================================
# CREATE RETRIEVERS
# ============================================================

dense_retriever = Retriever(
    embedder=embedder,
    vector_store=vector_store,
    chunks=chunks
)

bm25_retriever = BM25Retriever(
    chunks=chunks
)

hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    bm25_retriever=bm25_retriever
)


# ============================================================
# CREATE RERANKER
# ============================================================

reranker = Reranker()


# ============================================================
# CREATE EVIDENCE SUPPORT JUDGE
# ============================================================

judge = EvidenceSupportJudge()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def first_relevant_rank(
    results: list[dict],
    relevant_chunks: list[int]
):
    """
    Return the first rank at which a gold-relevant chunk
    appears.

    This is evaluation-only diagnostic information.
    """

    relevant_chunks = set(relevant_chunks)

    for rank, result in enumerate(
        results,
        start=1
    ):
        if result["chunk_index"] in relevant_chunks:
            return rank

    return None


def relevant_chunks_found(
    results: list[dict],
    relevant_chunks: list[int]
):
    """
    Return gold-relevant chunks that appear
    in the retrieved results.

    Evaluation-only.
    """

    relevant_chunks = set(relevant_chunks)

    return [
        result["chunk_index"]
        for result in results
        if result["chunk_index"] in relevant_chunks
    ]


def classify_failure(
    expected_behavior: str,
    predicted_behavior: str,
    top_50_results: list[dict],
    top_10_results: list[dict],
    relevant_chunks: list[int]
):
    """
    Classify the observed failure.

    IMPORTANT:
    relevant_chunks are used ONLY for post-hoc evaluation.
    They are never passed to the judge.
    """

    top_50_relevant = relevant_chunks_found(
        top_50_results,
        relevant_chunks
    )

    top_10_relevant = relevant_chunks_found(
        top_10_results,
        relevant_chunks
    )

    # --------------------------------------------------------
    # CASE 1: Correct behavior
    # --------------------------------------------------------

    if expected_behavior == predicted_behavior:
        return None


    # --------------------------------------------------------
    # CASE 2: Expected answer but system abstained
    # --------------------------------------------------------

    if (
        expected_behavior == "answer"
        and predicted_behavior == "abstain"
    ):

        # No relevant evidence anywhere in Top-50
        # => retrieval failure

        if not top_50_relevant:
            return "retrieval_failure"

        # Relevant evidence exists in Top-50 but not Top-10
        # => ranking failure

        if not top_10_relevant:
            return "ranking_failure"

        # Relevant evidence reached Top-10
        # but judge rejected it
        # => judge failure

        return "judge_failure"


    # --------------------------------------------------------
    # CASE 3: Expected abstention but system answered
    # --------------------------------------------------------

    if (
        expected_behavior == "abstain"
        and predicted_behavior == "answer"
    ):

        return "unsupported_acceptance"


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return "unknown_failure"


def build_evidence_records(
    results: list[dict],
    relevant_chunks: list[int]
):
    """
    Save detailed evidence information.

    This information is stored in JSON but is NOT printed
    fully to the terminal.
    """

    relevant_chunks = set(relevant_chunks)

    evidence_records = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        chunk_index = result["chunk_index"]

        evidence_records.append(
            {
                "rank": rank,
                "chunk_index": chunk_index,
                "is_gold_relevant": (
                    chunk_index in relevant_chunks
                ),
                "retrieval_score": result.get(
                    "score",
                    0.0
                ),
                "reranker_score": result.get(
                    "reranker_score",
                    0.0
                ),
                "text": result["text"]
            }
        )

    return evidence_records


# ============================================================
# COMPACT TERMINAL DIAGNOSTIC
# ============================================================

def print_diagnostic(
    question_data: dict,
    failure_type: str,
    expected_behavior: str,
    predicted_behavior: str,
    top_50_results: list[dict],
    final_results: list[dict],
    judge_result: dict
):

    question_id = question_data["id"]

    question = question_data["question"]

    relevant_chunks = question_data.get(
        "relevant_chunks",
        []
    )

    top_50_ids = [
        result["chunk_index"]
        for result in top_50_results
    ]

    top_10_ids = [
        result["chunk_index"]
        for result in final_results
    ]

    first_rank_50 = first_relevant_rank(
        top_50_results,
        relevant_chunks
    )

    first_rank_10 = first_relevant_rank(
        final_results,
        relevant_chunks
    )

    print()
    print("=" * 80)

    print(
        f"Q{question_id} | {failure_type}"
    )

    print(
        f"Question: {question}"
    )

    print(
        f"Expected: {expected_behavior} | "
        f"Predicted: {predicted_behavior}"
    )

    print(
        f"Gold relevant chunks: {relevant_chunks}"
    )

    print(
        f"First relevant rank Top-50: "
        f"{first_rank_50}"
    )

    print(
        f"First relevant rank Top-10: "
        f"{first_rank_10}"
    )

    print(
        f"Top-50 chunks: {top_50_ids}"
    )

    print(
        f"Top-10 chunks: {top_10_ids}"
    )

    print(
        f"Judge raw output: "
        f"{judge_result['raw_output']}"
    )

    print()
    print("Top-10 evidence summary:")

    relevant_set = set(relevant_chunks)

    for rank, result in enumerate(
        final_results,
        start=1
    ):

        chunk_index = result["chunk_index"]

        is_relevant = (
            chunk_index in relevant_set
        )

        marker = "*" if is_relevant else " "

        reranker_score = result.get(
            "reranker_score",
            0.0
        )

        retrieval_score = result.get(
            "score",
            0.0
        )

        print(
            f"{marker} Rank {rank:02d} | "
            f"Chunk {chunk_index:02d} | "
            f"Retrieval: {retrieval_score:.4f} | "
            f"Reranker: {reranker_score:.4f}"
        )

    print()
    print(
        "* = gold-relevant chunk "
        "(evaluation-only)"
    )

    print("=" * 80)


# ============================================================
# MAIN EVALUATION
# ============================================================

failures = []

total_answerable = 0
total_unanswerable = 0

correct_answerable = 0
correct_unanswerable = 0


for question_data in questions:

    question_id = question_data["id"]

    question = question_data["question"]

    answerability = question_data[
        "answerability"
    ]

    relevant_chunks = question_data.get(
        "relevant_chunks",
        []
    )


    # --------------------------------------------------------
    # Expected behavior
    # --------------------------------------------------------

    if answerability == "answerable":

        expected_behavior = "answer"

        total_answerable += 1

    else:

        expected_behavior = "abstain"

        total_unanswerable += 1


    # --------------------------------------------------------
    # Retrieve Top-50 candidate pool
    # --------------------------------------------------------

    top_50_results = hybrid_retriever.retrieve(
        query=question,
        top_k=CANDIDATE_K,
        candidate_k=CANDIDATE_K
    )


    # --------------------------------------------------------
    # Rerank candidate pool
    # --------------------------------------------------------

    reranked_results = reranker.rerank(
        question,
        top_50_results
    )


    # --------------------------------------------------------
    # Select final Top-10 evidence
    # --------------------------------------------------------

    final_results = reranked_results[
        :TOP_K
    ]


    # --------------------------------------------------------
    # Build evidence text
    #
    # IMPORTANT:
    # Gold answers/relevant chunks are NOT used here.
    # --------------------------------------------------------

    evidence = "\n\n".join(
        result["text"]
        for result in final_results
    )


    # --------------------------------------------------------
    # Evidence support judgment
    # --------------------------------------------------------

    judge_result = judge.judge(
        question=question,
        evidence=evidence
    )

    predicted_label = judge_result[
        "label"
    ]


    if predicted_label == "SUPPORTED":

        predicted_behavior = "answer"

    else:

        predicted_behavior = "abstain"


    # --------------------------------------------------------
    # Track correct behavior
    # --------------------------------------------------------

    if (
        expected_behavior
        == predicted_behavior
    ):

        if answerability == "answerable":

            correct_answerable += 1

        else:

            correct_unanswerable += 1

        continue


    # --------------------------------------------------------
    # Classify failure
    # --------------------------------------------------------

    failure_type = classify_failure(
        expected_behavior=expected_behavior,
        predicted_behavior=predicted_behavior,
        top_50_results=top_50_results,
        top_10_results=final_results,
        relevant_chunks=relevant_chunks
    )


    # --------------------------------------------------------
    # Build detailed failure record
    # --------------------------------------------------------

    failure_record = {

        "id": question_id,

        "question": question,

        "answerability": answerability,

        "expected_behavior": expected_behavior,

        "predicted_behavior": predicted_behavior,

        "failure_type": failure_type,

        # Evaluation-only gold information
        "relevant_chunks": relevant_chunks,

        "top_50_chunk_ids": [
            result["chunk_index"]
            for result in top_50_results
        ],

        "first_relevant_rank_top_50":
            first_relevant_rank(
                top_50_results,
                relevant_chunks
            ),

        "top_10_chunk_ids": [
            result["chunk_index"]
            for result in final_results
        ],

        "first_relevant_rank_top_10":
            first_relevant_rank(
                final_results,
                relevant_chunks
            ),

        # Complete evidence for later inspection
        "top_10_evidence":
            build_evidence_records(
                final_results,
                relevant_chunks
            ),

        "raw_output":
            judge_result["raw_output"]
    }


    failures.append(
        failure_record
    )


    # --------------------------------------------------------
    # Print compact diagnostic
    # --------------------------------------------------------

    print_diagnostic(
        question_data=question_data,
        failure_type=failure_type,
        expected_behavior=expected_behavior,
        predicted_behavior=predicted_behavior,
        top_50_results=top_50_results,
        final_results=final_results,
        judge_result=judge_result
    )


# ============================================================
# FAILURE SUMMARY
# ============================================================

failure_counts = {

    "retrieval_failure": 0,

    "ranking_failure": 0,

    "judge_failure": 0,

    "unsupported_acceptance": 0,

    "unknown_failure": 0
}


for failure in failures:

    failure_type = failure[
        "failure_type"
    ]

    if failure_type in failure_counts:

        failure_counts[
            failure_type
        ] += 1


# ============================================================
# SUMMARY IDS
# ============================================================

failure_ids = {

    "retrieval_failure": [],

    "ranking_failure": [],

    "judge_failure": [],

    "unsupported_acceptance": [],

    "unknown_failure": []
}


for failure in failures:

    failure_type = failure[
        "failure_type"
    ]

    failure_ids[
        failure_type
    ].append(
        failure["id"]
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

total_questions = len(
    questions
)

total_correct = (
    correct_answerable
    + correct_unanswerable
)

behavior_accuracy = (
    total_correct
    / total_questions
)


summary = {

    "total_questions":
        total_questions,

    "answerable_questions":
        total_answerable,

    "unanswerable_questions":
        total_unanswerable,

    "correct_answerable":
        correct_answerable,

    "correct_unanswerable":
        correct_unanswerable,

    "total_correct":
        total_correct,

    "behavior_accuracy":
        behavior_accuracy,

    "total_failures":
        len(failures),

    "failure_counts":
        failure_counts,

    "failure_ids":
        failure_ids
}


# ============================================================
# SAVE RESULTS
# ============================================================

output = {

    "config": {

        "top_k":
            TOP_K,

        "candidate_k":
            CANDIDATE_K,

        "chunk_size":
            CHUNK_SIZE,

        "chunk_overlap":
            CHUNK_OVERLAP,

        "questions_path":
            QUESTIONS_PATH,

        "pdf_path":
            PDF_PATH
    },

    "summary":
        summary,

    "failures":
        failures
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
# FINAL TERMINAL SUMMARY
# ============================================================

print()
print()
print("=" * 80)
print("FAILURE SUMMARY")
print("=" * 80)

print(
    f"Total questions: "
    f"{total_questions}"
)

print(
    f"Total failures: "
    f"{len(failures)}"
)

print()

print(
    f"Retrieval failures: "
    f"{failure_counts['retrieval_failure']}"
)

print(
    f"Ranking failures: "
    f"{failure_counts['ranking_failure']}"
)

print(
    f"Judge failures: "
    f"{failure_counts['judge_failure']}"
)

print(
    f"Unsupported acceptance: "
    f"{failure_counts['unsupported_acceptance']}"
)

print()

print(
    f"Correct answerable: "
    f"{correct_answerable}/{total_answerable}"
)

print(
    f"Correct unanswerable: "
    f"{correct_unanswerable}/{total_unanswerable}"
)

print(
    f"Overall behavior accuracy: "
    f"{behavior_accuracy:.3f}"
)

print()
print("Failures by category:")

for failure_type, ids in failure_ids.items():

    print(
        f"{failure_type}: {ids}"
    )

print()
print(
    f"Saved diagnostic results to:"
)

print(
    OUTPUT_PATH
)

print("=" * 80)

