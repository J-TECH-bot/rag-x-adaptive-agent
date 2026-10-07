from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text

from embeddings.embedder import Embedder

from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

from evaluation.evidence_support_v2 import EvidenceSupportJudgeV2

from adaptive.decision_engine import AdaptiveDecisionEngine
from adaptive.evidence_selector import AdaptiveEvidenceSelector
from adaptive.adaptive_retriever import AdaptiveRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rag_original.pdf"
)

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_questions.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "adaptive_rag_results.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "adaptive_rag_summary.json"
)


# ==============================================================
# DATASET
# ==============================================================

def load_questions():
    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        questions = json.load(f)

    if len(questions) != 60:
        raise ValueError(
            f"Expected 60 questions, found {len(questions)}"
        )

    required_fields = {
        "id",
        "question",
        "answerability",
        "difficulty_type",
        "relevant_chunks",
        "reference_answer",
    }

    for question in questions:
        missing = required_fields - set(question)

        if missing:
            raise ValueError(
                f"Question {question.get('id')} "
                f"missing fields: {sorted(missing)}"
            )

    return questions


# ==============================================================
# PIPELINE
# ==============================================================

def build_system(chunks):
    print()
    print("=" * 80)
    print("BUILDING RAG-X ADAPTIVE SYSTEM")
    print("=" * 80)

    print("\nLoading embedding model...")

    embedder = Embedder(
        model_name="all-MiniLM-L6-v2"
    )

    print("Embedding document chunks...")

    embeddings = embedder.embed(chunks)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(embeddings)

    dense_retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks,
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    print("\nLoading reranker...")

    reranker = Reranker()

    print("\nLoading evidence judge...")

    evidence_judge = EvidenceSupportJudgeV2()

    decision_engine = AdaptiveDecisionEngine(
        max_retrieval_rounds=2
    )

    evidence_selector = AdaptiveEvidenceSelector(
        evidence_judge=evidence_judge,
        evidence_depths=(1, 3, 5, 10),
    )

    adaptive_retriever = AdaptiveRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        evidence_judge=evidence_judge,
        decision_engine=decision_engine,
        evidence_selector=evidence_selector,
        initial_candidate_k=20,
        expanded_candidate_k=50,
    )

    print("\nAdaptive system ready.")

    return adaptive_retriever


# ==============================================================
# HELPERS
# ==============================================================

def normalize(value):
    if value is None:
        return ""

    return str(value).strip().upper()


def candidate_indices(candidates):
    return [
        int(item["chunk_index"])
        for item in candidates
    ]


def first_relevant_rank(
    candidates,
    relevant_chunks,
):
    relevant = {
        int(chunk)
        for chunk in relevant_chunks
    }

    if not relevant:
        return None

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        if int(candidate["chunk_index"]) in relevant:
            return rank

    return None


def hit_at_k(
    candidates,
    relevant_chunks,
    k,
):
    relevant = {
        int(chunk)
        for chunk in relevant_chunks
    }

    if not relevant:
        return None

    top_k = candidates[:k]

    return any(
        int(item["chunk_index"]) in relevant
        for item in top_k
    )


def get_retrieval_metrics(
    candidates,
    relevant_chunks,
):
    if not relevant_chunks:
        return {
            "gold_chunks": 0,
            "first_relevant_rank": None,
            "hit_at_1": None,
            "hit_at_3": None,
            "hit_at_5": None,
            "hit_at_10": None,
            "hit_at_20": None,
            "hit_at_50": None,
        }

    return {
        "gold_chunks": len(relevant_chunks),
        "first_relevant_rank": first_relevant_rank(
            candidates,
            relevant_chunks,
        ),
        "hit_at_1": hit_at_k(
            candidates,
            relevant_chunks,
            1,
        ),
        "hit_at_3": hit_at_k(
            candidates,
            relevant_chunks,
            3,
        ),
        "hit_at_5": hit_at_k(
            candidates,
            relevant_chunks,
            5,
        ),
        "hit_at_10": hit_at_k(
            candidates,
            relevant_chunks,
            10,
        ),
        "hit_at_20": hit_at_k(
            candidates,
            relevant_chunks,
            20,
        ),
        "hit_at_50": hit_at_k(
            candidates,
            relevant_chunks,
            50,
        ),
    }


def classify_failure(
    answerability,
    final_decision,
    relevant_chunks,
    trace,
):
    """
    Diagnose why an answerable question failed.

    This is deliberately based on the actual retrieval/reranking
    trace rather than the evidence judge alone.
    """

    if answerability != "answerable":
        return None

    if final_decision == "ANSWER":
        return "none"

    relevant = {
        int(chunk)
        for chunk in relevant_chunks
    }

    if not relevant:
        return "no_gold_evidence"

    # ----------------------------------------------------------
    # Check every retrieval round.
    #
    # If a relevant chunk appears in retrieved_candidates,
    # retrieval succeeded at that round.
    # ----------------------------------------------------------
    retrieved_any = False
    reranked_any = False

    best_reranked_rank = None

    for round_data in trace:
        retrieved = round_data.get(
            "retrieved_candidates",
            []
        )

        reranked = round_data.get(
            "reranked_candidates",
            []
        )

        retrieved_indices = {
            int(item["chunk_index"])
            for item in retrieved
        }

        reranked_indices = {
            int(item["chunk_index"])
            for item in reranked
        }

        if relevant.intersection(
            retrieved_indices
        ):
            retrieved_any = True

        if relevant.intersection(
            reranked_indices
        ):
            reranked_any = True

            rank = first_relevant_rank(
                reranked,
                relevant_chunks,
            )

            if rank is not None:
                if (
                    best_reranked_rank is None
                    or rank < best_reranked_rank
                ):
                    best_reranked_rank = rank

    # ----------------------------------------------------------
    # Relevant chunk never entered any candidate set.
    # ----------------------------------------------------------
    if not retrieved_any:
        return "retrieval_failure"

    # ----------------------------------------------------------
    # Relevant chunk was retrieved but never appeared in
    # reranked candidates.
    #
    # This should rarely happen because reranker normally
    # returns every candidate it receives.
    # ----------------------------------------------------------
    if not reranked_any:
        return "ranking_failure"

    # ----------------------------------------------------------
    # Relevant chunk reached the reranked list but did not
    # produce a supported answer.
    #
    # We distinguish:
    #
    # <= 10  -> evidence/judge failure
    # > 10   -> ranking depth failure
    # ----------------------------------------------------------
    if best_reranked_rank is not None:
        if best_reranked_rank <= 10:
            return "evidence_sufficiency_failure"

        return "ranking_depth_failure"

    return "unknown_failure"


def extract_round_statistics(trace):
    evidence_depths_checked = []

    for round_data in trace:
        for evidence_round in round_data.get(
            "evidence_rounds",
            [],
        ):
            depth = evidence_round.get(
                "depth"
            )

            if depth is not None:
                evidence_depths_checked.append(
                    int(depth)
                )

    return evidence_depths_checked


# ==============================================================
# SINGLE QUESTION
# ==============================================================

def evaluate_question(
    adaptive_retriever,
    question_data,
    number,
    total,
):
    question_id = question_data["id"]

    question = question_data["question"]

    answerability = normalize(
        question_data["answerability"]
    ).lower()

    difficulty = question_data[
        "difficulty_type"
    ]

    relevant_chunks = question_data[
        "relevant_chunks"
    ]

    print()
    print(
        "=" * 80
    )

    print(
        f"[{number}/{total}] "
        f"Q{question_id}: {question}"
    )

    print(
        f"Expected: {answerability} | "
        f"Difficulty: {difficulty}"
    )

    result = adaptive_retriever.retrieve(
        query=question
    )

    final_decision = result[
        "decision"
    ]

    final_status = (
        "SUPPORTED"
        if final_decision == "ANSWER"
        else "ABSTAIN"
    )

    trace = result.get(
        "trace",
        []
    )

    final_trace = (
        trace[-1]
        if trace
        else {}
    )

    first_trace = (
        trace[0]
        if trace
        else {}
    )

    # ----------------------------------------------------------
    # Outcome
    # ----------------------------------------------------------

    if answerability == "answerable":

        if final_decision == "ANSWER":
            outcome = "SUPPORTED_ANSWERABLE"
        else:
            outcome = "MISSED_ANSWERABLE"

    else:

        if final_decision == "ABSTAIN":
            outcome = "CORRECT_ABSTENTION"
        else:
            outcome = "FALSE_SUPPORT"

    # ----------------------------------------------------------
    # Retrieval metrics
    # ----------------------------------------------------------

    final_reranked = final_trace.get(
        "reranked_candidates",
        []
    )

    final_retrieved = final_trace.get(
        "retrieved_candidates",
        []
    )

    final_retrieval_metrics = (
        get_retrieval_metrics(
            final_reranked,
            relevant_chunks,
        )
    )

    # ----------------------------------------------------------
    # Also record candidate-set reachability across rounds.
    # ----------------------------------------------------------

    round_metrics = []

    for round_data in trace:

        retrieved = round_data.get(
            "retrieved_candidates",
            []
        )

        reranked = round_data.get(
            "reranked_candidates",
            []
        )

        round_metrics.append(
            {
                "retrieval_round": round_data[
                    "retrieval_round"
                ],
                "candidate_k": round_data[
                    "candidate_k"
                ],
                "candidate_count": round_data[
                    "candidate_count"
                ],

                "retrieval_metrics": (
                    get_retrieval_metrics(
                        retrieved,
                        relevant_chunks,
                    )
                ),

                "reranking_metrics": (
                    get_retrieval_metrics(
                        reranked,
                        relevant_chunks,
                    )
                ),
            }
        )

    # ----------------------------------------------------------
    # Failure diagnosis
    # ----------------------------------------------------------

    failure_type = classify_failure(
        answerability=answerability,
        final_decision=final_decision,
        relevant_chunks=relevant_chunks,
        trace=trace,
    )

    # ----------------------------------------------------------
    # Efficiency
    # ----------------------------------------------------------

    evidence_depths_checked = (
        extract_round_statistics(
            trace
        )
    )

    selected_evidence_depth = result[
        "selected_evidence_depth"
    ]

    retrieval_round_count = len(
        trace
    )

    evidence_round_count = len(
        evidence_depths_checked
    )

    expanded = (
        retrieval_round_count > 1
    )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------

    print(
        f"Decision: {final_decision}"
    )

    print(
        f"Outcome: {outcome}"
    )

    print(
        f"Candidate depth used: "
        f"{final_trace.get('candidate_k')}"
    )

    print(
        f"Candidate count: "
        f"{final_trace.get('candidate_count')}"
    )

    print(
        f"Selected evidence depth: "
        f"{selected_evidence_depth}"
    )

    print(
        f"Retrieval rounds: "
        f"{retrieval_round_count}"
    )

    print(
        f"Retrieval expanded: "
        f"{expanded}"
    )

    if failure_type:
        print(
            f"Failure diagnosis: "
            f"{failure_type}"
        )

    print(
        f"Reason: "
        f"{result['reason']}"
    )

    return {
        "id": question_id,
        "question": question,

        "answerability": question_data[
            "answerability"
        ],

        "difficulty_type": difficulty,

        "relevant_chunks": relevant_chunks,

        "reference_answer": question_data[
            "reference_answer"
        ],

        "predicted_decision": final_decision,

        "predicted_status": final_status,

        "evaluation_outcome": outcome,

        "failure_type": failure_type,

        "selected_evidence_depth": (
            selected_evidence_depth
        ),

        "final_candidate_depth": (
            final_trace.get(
                "candidate_k"
            )
        ),

        "final_candidate_count": (
            final_trace.get(
                "candidate_count"
            )
        ),

        "retrieval_rounds": (
            retrieval_round_count
        ),

        "evidence_rounds": (
            evidence_round_count
        ),

        "evidence_depths_checked": (
            evidence_depths_checked
        ),

        "retrieval_expanded": expanded,

        "first_round_decision": (
            first_trace.get(
                "decision"
            )
        ),

        "final_evidence_label": (
            final_trace.get(
                "evidence_label"
            )
        ),

        "final_retrieval_metrics": (
            final_retrieval_metrics
        ),

        "round_metrics": round_metrics,

        "trace": trace,

        "selected_evidence": result.get(
            "evidence",
            []
        ),
    }


# ==============================================================
# AGGREGATION
# ==============================================================

def safe_mean(values):
    if not values:
        return 0.0

    return round(
        statistics.mean(values),
        3,
    )


def rate(count, total):
    if total == 0:
        return 0.0

    return round(
        count / total,
        3,
    )


def build_hit_summary(
    results,
    metric_prefix,
):
    output = {}

    for k in [
        1,
        3,
        5,
        10,
        20,
        50,
    ]:
        key = f"hit_at_{k}"

        values = []

        for result in results:

            metrics = result.get(
                metric_prefix,
                {}
            )

            value = metrics.get(
                key
            )

            if value is not None:
                values.append(
                    bool(value)
                )

        output[key] = {
            "hits": sum(values),
            "total": len(values),
            "rate": rate(
                sum(values),
                len(values),
            ),
        }

    return output


def build_summary(results):
    total = len(results)

    answerable = [
        r
        for r in results
        if r["answerability"].lower()
        == "answerable"
    ]

    unanswerable = [
        r
        for r in results
        if r["answerability"].lower()
        == "unanswerable"
    ]

    supported = [
        r
        for r in results
        if r["evaluation_outcome"]
        == "SUPPORTED_ANSWERABLE"
    ]

    missed = [
        r
        for r in results
        if r["evaluation_outcome"]
        == "MISSED_ANSWERABLE"
    ]

    correct_abstentions = [
        r
        for r in results
        if r["evaluation_outcome"]
        == "CORRECT_ABSTENTION"
    ]

    false_supports = [
        r
        for r in results
        if r["evaluation_outcome"]
        == "FALSE_SUPPORT"
    ]

    expanded = [
        r
        for r in results
        if r["retrieval_expanded"]
    ]

    solved_at_20 = [
        r
        for r in supported
        if r["first_round_decision"]
        == "ANSWER"
    ]

    selected_depths = [
        r["selected_evidence_depth"]
        for r in results
        if r["selected_evidence_depth"] > 0
    ]

    candidate_depths = [
        r["final_candidate_depth"]
        for r in results
        if r["final_candidate_depth"]
    ]

    retrieval_rounds = [
        r["retrieval_rounds"]
        for r in results
    ]

    evidence_rounds = [
        r["evidence_rounds"]
        for r in results
    ]

    failure_distribution = Counter(
        r["failure_type"]
        for r in results
        if r["failure_type"]
    )

    outcome_distribution = Counter(
        r["evaluation_outcome"]
        for r in results
    )

    evidence_distribution = Counter(
        r["selected_evidence_depth"]
        for r in results
    )

    candidate_distribution = Counter(
        r["final_candidate_depth"]
        for r in results
    )

    difficulty_summary = {}

    difficulties = sorted(
        set(
            r["difficulty_type"]
            for r in results
        )
    )

    for difficulty in difficulties:

        subset = [
            r
            for r in results
            if r["difficulty_type"]
            == difficulty
        ]

        subset_supported = sum(
            r["evaluation_outcome"]
            == "SUPPORTED_ANSWERABLE"
            for r in subset
        )

        subset_missed = sum(
            r["evaluation_outcome"]
            == "MISSED_ANSWERABLE"
            for r in subset
        )

        subset_correct_abstention = sum(
            r["evaluation_outcome"]
            == "CORRECT_ABSTENTION"
            for r in subset
        )

        subset_false_support = sum(
            r["evaluation_outcome"]
            == "FALSE_SUPPORT"
            for r in subset
        )

        difficulty_summary[difficulty] = {
            "total": len(subset),

            "supported_answerable": (
                subset_supported
            ),

            "missed_answerable": (
                subset_missed
            ),

            "correct_abstention": (
                subset_correct_abstention
            ),

            "false_support": (
                subset_false_support
            ),

            "support_rate": rate(
                subset_supported,
                sum(
                    r["answerability"].lower()
                    == "answerable"
                    for r in subset
                ),
            ),

            "average_evidence_depth": safe_mean(
                [
                    r["selected_evidence_depth"]
                    for r in subset
                    if r["selected_evidence_depth"]
                ]
            ),

            "average_candidate_depth": safe_mean(
                [
                    r["final_candidate_depth"]
                    for r in subset
                    if r["final_candidate_depth"]
                ]
            ),
        }

    # ----------------------------------------------------------
    # Answerable retrieval/ranking analysis
    # ----------------------------------------------------------

    answerable_results = [
        r
        for r in results
        if r["answerability"].lower()
        == "answerable"
    ]

    final_reranking_results = [
        r
        for r in answerable_results
        if r["final_retrieval_metrics"].get(
            "hit_at_1"
        ) is not None
    ]

    final_retrieval_hits = build_hit_summary(
        final_reranking_results,
        "final_retrieval_metrics",
    )

    # ----------------------------------------------------------
    # Determine how many answerable questions had their gold
    # evidence reachable at candidate depth 20 and 50.
    # ----------------------------------------------------------

    reachable_at_20 = 0
    reachable_at_50 = 0

    reranked_top_10_at_20 = 0
    reranked_top_10_at_50 = 0

    for result in answerable_results:

        rounds = result["round_metrics"]

        for round_data in rounds:

            metrics = round_data[
                "retrieval_metrics"
            ]

            candidate_k = round_data[
                "candidate_k"
            ]

            if (
                candidate_k == 20
                and metrics["hit_at_20"]
            ):
                reachable_at_20 += 1

            if (
                candidate_k == 50
                and metrics["hit_at_50"]
            ):
                reachable_at_50 += 1

            reranking = round_data[
                "reranking_metrics"
            ]

            if (
                candidate_k == 20
                and reranking["hit_at_10"]
            ):
                reranked_top_10_at_20 += 1

            if (
                candidate_k == 50
                and reranking["hit_at_10"]
            ):
                reranked_top_10_at_50 += 1

    summary = {
        "experiment": "RAG-X Adaptive RAG Evaluation",

        "dataset": {
            "total": total,
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
        },

        "outcomes": {
            "supported_answerable": len(
                supported
            ),

            "missed_answerable": len(
                missed
            ),

            "correct_abstentions": len(
                correct_abstentions
            ),

            "false_supports": len(
                false_supports
            ),
        },

        "rates": {
            "answerable_support_rate": rate(
                len(supported),
                len(answerable),
            ),

            "answerable_miss_rate": rate(
                len(missed),
                len(answerable),
            ),

            "unanswerable_abstention_rate": rate(
                len(correct_abstentions),
                len(unanswerable),
            ),

            "false_support_rate": rate(
                len(false_supports),
                len(unanswerable),
            ),

            "overall_correct_rate": rate(
                len(supported)
                + len(correct_abstentions),
                total,
            ),
        },

        "efficiency": {
            "average_selected_evidence_depth": (
                safe_mean(
                    selected_depths
                )
            ),

            "average_final_candidate_depth": (
                safe_mean(
                    candidate_depths
                )
            ),

            "average_retrieval_rounds": (
                safe_mean(
                    retrieval_rounds
                )
            ),

            "average_evidence_rounds": (
                safe_mean(
                    evidence_rounds
                )
            ),

            "solved_at_candidate_20": (
                len(solved_at_20)
            ),

            "retrieval_expansions": (
                len(expanded)
            ),

            "retrieval_expansion_rate": rate(
                len(expanded),
                total,
            ),
        },

        "evidence_depth_distribution": {
            str(k): v
            for k, v in sorted(
                evidence_distribution.items()
            )
        },

        "candidate_depth_distribution": {
            str(k): v
            for k, v in sorted(
                candidate_distribution.items()
            )
        },

        "outcome_distribution": {
            str(k): v
            for k, v in sorted(
                outcome_distribution.items()
            )
        },

        "failure_distribution": {
            str(k): v
            for k, v in sorted(
                failure_distribution.items()
            )
        },

        "retrieval_reachability": {
            "answerable_relevant_reached_at_candidate_20": (
                reachable_at_20
            ),

            "answerable_relevant_reached_at_candidate_50": (
                reachable_at_50
            ),

            "answerable_total": len(
                answerable
            ),

            "reachability_rate_at_20": rate(
                reachable_at_20,
                len(answerable),
            ),

            "reachability_rate_at_50": rate(
                reachable_at_50,
                len(answerable),
            ),
        },

        "reranking_reachability": {
            "answerable_relevant_in_reranked_top_10_at_20": (
                reranked_top_10_at_20
            ),

            "answerable_relevant_in_reranked_top_10_at_50": (
                reranked_top_10_at_50
            ),

            "answerable_total": len(
                answerable
            ),

            "top_10_rate_at_20": rate(
                reranked_top_10_at_20,
                len(answerable),
            ),

            "top_10_rate_at_50": rate(
                reranked_top_10_at_50,
                len(answerable),
            ),
        },

        "final_reranked_hit_metrics": (
            final_retrieval_hits
        ),

        "difficulty_summary": (
            difficulty_summary
        ),

        "dataset_answerability_counts": dict(
            Counter(
                r["answerability"]
                for r in results
            )
        ),

        "dataset_difficulty_counts": dict(
            Counter(
                r["difficulty_type"]
                for r in results
            )
        ),
    }

    return summary


# ==============================================================
# MAIN
# ==============================================================

def main():

    print()
    print("=" * 80)
    print("RAG-X ADAPTIVE RAG — 60 QUESTION EVALUATION")
    print("=" * 80)

    questions = load_questions()

    print(
        f"\nQuestions: {len(questions)}"
    )

    answerability_counts = Counter(
        q["answerability"]
        for q in questions
    )

    difficulty_counts = Counter(
        q["difficulty_type"]
        for q in questions
    )

    print("\nAnswerability:")

    for key, value in answerability_counts.items():
        print(
            f"  {key}: {value}"
        )

    print("\nDifficulty:")

    for key, value in difficulty_counts.items():
        print(
            f"  {key}: {value}"
        )

    print("\nLoading PDF...")

    text = load_pdf(
        str(PDF_PATH)
    )

    if not text.strip():
        raise ValueError(
            "PDF extraction returned empty text."
        )

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    adaptive_retriever = build_system(
        chunks
    )

    results = []

    print()
    print("=" * 80)
    print("RUNNING 60-QUESTION EVALUATION")
    print("=" * 80)

    for number, question_data in enumerate(
        questions,
        start=1,
    ):

        result = evaluate_question(
            adaptive_retriever=adaptive_retriever,
            question_data=question_data,
            number=number,
            total=len(questions),
        )

        results.append(
            result
        )

    summary = build_summary(
        results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ==========================================================
    # FINAL REPORT
    # ==========================================================

    print()
    print("=" * 80)
    print("FINAL RAG-X RESULTS")
    print("=" * 80)

    print("\nDataset:")

    print(
        f"  Total: "
        f"{summary['dataset']['total']}"
    )

    print(
        f"  Answerable: "
        f"{summary['dataset']['answerable']}"
    )

    print(
        f"  Unanswerable: "
        f"{summary['dataset']['unanswerable']}"
    )

    print("\nOutcomes:")

    print(
        f"  Supported answerable: "
        f"{summary['outcomes']['supported_answerable']}"
    )

    print(
        f"  Missed answerable: "
        f"{summary['outcomes']['missed_answerable']}"
    )

    print(
        f"  Correct abstentions: "
        f"{summary['outcomes']['correct_abstentions']}"
    )

    print(
        f"  False supports: "
        f"{summary['outcomes']['false_supports']}"
    )

    print("\nRates:")

    print(
        f"  Answerable support rate: "
        f"{summary['rates']['answerable_support_rate']:.1%}"
    )

    print(
        f"  Unanswerable abstention rate: "
        f"{summary['rates']['unanswerable_abstention_rate']:.1%}"
    )

    print(
        f"  False support rate: "
        f"{summary['rates']['false_support_rate']:.1%}"
    )

    print(
        f"  Overall correct rate: "
        f"{summary['rates']['overall_correct_rate']:.1%}"
    )

    print("\nEfficiency:")

    print(
        f"  Average selected evidence depth: "
        f"{summary['efficiency']['average_selected_evidence_depth']}"
    )

    print(
        f"  Average final candidate depth: "
        f"{summary['efficiency']['average_final_candidate_depth']}"
    )

    print(
        f"  Average retrieval rounds: "
        f"{summary['efficiency']['average_retrieval_rounds']}"
    )

    print(
        f"  Average evidence rounds: "
        f"{summary['efficiency']['average_evidence_rounds']}"
    )

    print(
        f"  Solved at candidate-20: "
        f"{summary['efficiency']['solved_at_candidate_20']}"
    )

    print(
        f"  Retrieval expansions: "
        f"{summary['efficiency']['retrieval_expansions']}"
    )

    print("\nEvidence depth distribution:")

    for depth, count in summary[
        "evidence_depth_distribution"
    ].items():

        print(
            f"  Top-{depth}: {count}"
        )

    print("\nCandidate depth distribution:")

    for depth, count in summary[
        "candidate_depth_distribution"
    ].items():

        print(
            f"  Candidate-{depth}: {count}"
        )

    print("\nFailure distribution:")

    for failure, count in summary[
        "failure_distribution"
    ].items():

        print(
            f"  {failure}: {count}"
        )

    print("\nRetrieval reachability:")

    reachability = summary[
        "retrieval_reachability"
    ]

    print(
        f"  Relevant evidence reached at candidate-20: "
        f"{reachability['answerable_relevant_reached_at_candidate_20']}/"
        f"{reachability['answerable_total']}"
    )

    print(
        f"  Relevant evidence reached at candidate-50: "
        f"{reachability['answerable_relevant_reached_at_candidate_50']}/"
        f"{reachability['answerable_total']}"
    )

    print("\nReranking reachability:")

    reranking = summary[
        "reranking_reachability"
    ]

    print(
        f"  Relevant evidence in reranked Top-10 at candidate-20: "
        f"{reranking['answerable_relevant_in_reranked_top_10_at_20']}/"
        f"{reranking['answerable_total']}"
    )

    print(
        f"  Relevant evidence in reranked Top-10 at candidate-50: "
        f"{reranking['answerable_relevant_in_reranked_top_10_at_50']}/"
        f"{reranking['answerable_total']}"
    )

    print("\nDifficulty summary:")

    for difficulty, metrics in summary[
        "difficulty_summary"
    ].items():

        print()
        print(
            f"  {difficulty}"
        )

        print(
            f"    Total: "
            f"{metrics['total']}"
        )

        print(
            f"    Supported: "
            f"{metrics['supported_answerable']}"
        )

        print(
            f"    Missed: "
            f"{metrics['missed_answerable']}"
        )

        print(
            f"    Correct abstention: "
            f"{metrics['correct_abstention']}"
        )

        print(
            f"    False support: "
            f"{metrics['false_support']}"
        )

        print(
            f"    Support rate: "
            f"{metrics['support_rate']:.1%}"
        )

        print(
            f"    Avg evidence depth: "
            f"{metrics['average_evidence_depth']}"
        )

        print(
            f"    Avg candidate depth: "
            f"{metrics['average_candidate_depth']}"
        )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        f"\nDetailed results:\n"
        f"{RESULTS_PATH}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_PATH}"
    )

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
