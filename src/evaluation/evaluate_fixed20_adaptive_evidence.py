from __future__ import annotations

import json
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


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"

RESULTS_PATH = "data/evaluation/fixed20_adaptive_evidence_results.json"
SUMMARY_PATH = "data/evaluation/fixed20_adaptive_evidence_summary.json"

CANDIDATE_K = 20
EVIDENCE_DEPTHS = (1, 3, 5, 10)


def combine_evidence(results: list[dict]) -> str:
    return "\n\n".join(
        result["text"]
        for result in results
    )


def has_relevant_evidence(
    retrieved_results: list[dict],
    relevant_chunks: list[int],
) -> bool:

    retrieved_indices = {
        result["chunk_index"]
        for result in retrieved_results
    }

    return bool(
        retrieved_indices.intersection(
            relevant_chunks
        )
    )


def calculate_failure_type(
    decision: str,
    answerability: str,
    relevant_reached: bool,
    evidence_label: str,
) -> str:

    if answerability == "unanswerable":
        if decision == "ABSTAIN":
            return "none"
        return "unsupported_acceptance"

    if decision == "ANSWER":
        if relevant_reached:
            return "none"
        return "retrieval_or_ranking_failure"

    if not relevant_reached:
        return "retrieval_or_ranking_failure"

    if evidence_label == "INSUFFICIENT":
        return "evidence_support_failure"

    return "evidence_support_failure"


def main():

    print("=" * 70)
    print("FIXED TOP-20 CANDIDATE / ADAPTIVE EVIDENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        questions = json.load(f)

    total_questions = len(questions)

    answerable_count = sum(
        q["answerability"] == "answerable"
        for q in questions
    )

    unanswerable_count = sum(
        q["answerability"] == "unanswerable"
        for q in questions
    )

    print("\nDataset:")
    print(f"  Total: {total_questions}")
    print(f"  Answerable: {answerable_count}")
    print(f"  Unanswerable: {unanswerable_count}")

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    print("\nLoading PDF...")

    text = load_pdf(PDF_PATH)

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"  Chunks: {len(chunks)}")

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print("\nBuilding embeddings...")

    embedder = Embedder()

    embeddings = embedder.embed(chunks)

    print(f"  Embedding shape: {embeddings.shape}")

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(embeddings)

    dense_retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks,
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print("\nBuilding BM25 retriever...")

    bm25_retriever = BM25Retriever(chunks)

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    print("\nLoading reranker...")

    reranker = Reranker()

    # --------------------------------------------------------
    # Evidence judge
    # --------------------------------------------------------

    print("\nLoading evidence-support judge...")

    evidence_judge = EvidenceSupportJudgeV2()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    supported_answerable = 0
    missed_answerable = 0
    correct_abstentions = 0
    false_supports = 0

    retrieval_reached_count = 0
    reranking_reached_count = 0

    total_selected_evidence = 0
    total_evidence_rounds = 0

    evidence_depth_counter = Counter()
    failure_counter = Counter()
    difficulty_stats = {}

    print("\nRunning evaluation...")

    for i, item in enumerate(
        questions,
        start=1,
    ):

        question_id = item["id"]
        question = item["question"]
        answerability = item["answerability"]
        difficulty = item["difficulty_type"]
        relevant_chunks = item.get(
            "relevant_chunks",
            [],
        )

        # ----------------------------------------------------
        # FIXED CANDIDATE DEPTH = 20
        # ----------------------------------------------------

        retrieved = hybrid_retriever.retrieve(
            question,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        candidate_count = len(retrieved)

        # ----------------------------------------------------
        # RERANK ALL 20
        # ----------------------------------------------------

        reranked = reranker.rerank(
            question,
            retrieved,
        )

        # ----------------------------------------------------
        # REACHABILITY
        # ----------------------------------------------------

        relevant_reached = has_relevant_evidence(
            retrieved,
            relevant_chunks,
        )

        if answerability == "answerable":
            if relevant_reached:
                retrieval_reached_count += 1

            reranked_top10 = {
                result["chunk_index"]
                for result in reranked[:10]
            }

            if set(relevant_chunks).intersection(
                reranked_top10
            ):
                reranking_reached_count += 1

        # ----------------------------------------------------
        # ADAPTIVE EVIDENCE
        #
        # Try:
        #   Top-1
        #   Top-3
        #   Top-5
        #   Top-10
        #
        # Stop immediately when SUPPORTED.
        # ----------------------------------------------------

        rounds = []
        selected_evidence = []
        selected_depth = 0
        final_label = "UNKNOWN"
        decision = "ABSTAIN"

        for depth in EVIDENCE_DEPTHS:

            evidence = reranked[:depth]

            evidence_text = combine_evidence(
                evidence
            )

            judgment = evidence_judge.judge(
                question,
                evidence_text,
            )

            label = (
                judgment["label"]
                .strip()
                .upper()
            )

            rounds.append(
                {
                    "depth": depth,
                    "evidence_count": len(evidence),
                    "label": label,
                    "raw_output": judgment.get(
                        "raw_output",
                        "",
                    ),
                }
            )

            final_label = label
            selected_evidence = evidence
            selected_depth = min(
                depth,
                len(reranked),
            )

            total_evidence_rounds += 1

            if label == "SUPPORTED":
                decision = "ANSWER"
                break

        total_selected_evidence += selected_depth
        evidence_depth_counter[
            selected_depth
        ] += 1

        # ----------------------------------------------------
        # OUTCOME
        # ----------------------------------------------------

        if answerability == "answerable":

            if (
                decision == "ANSWER"
                and relevant_reached
            ):
                supported_answerable += 1
            else:
                missed_answerable += 1

        else:

            if decision == "ABSTAIN":
                correct_abstentions += 1
            else:
                false_supports += 1

        failure_type = calculate_failure_type(
            decision=decision,
            answerability=answerability,
            relevant_reached=relevant_reached,
            evidence_label=final_label,
        )

        failure_counter[
            failure_type
        ] += 1

        # ----------------------------------------------------
        # DIFFICULTY
        # ----------------------------------------------------

        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {
                "total": 0,
                "supported": 0,
                "correct_abstention": 0,
                "missed": 0,
                "false_support": 0,
            }

        difficulty_stats[difficulty][
            "total"
        ] += 1

        if answerability == "answerable":

            if (
                decision == "ANSWER"
                and relevant_reached
            ):
                difficulty_stats[difficulty][
                    "supported"
                ] += 1
            else:
                difficulty_stats[difficulty][
                    "missed"
                ] += 1

        else:

            if decision == "ABSTAIN":
                difficulty_stats[difficulty][
                    "correct_abstention"
                ] += 1
            else:
                difficulty_stats[difficulty][
                    "false_support"
                ] += 1

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        result = {
            "id": question_id,
            "question": question,
            "answerability": answerability,
            "difficulty_type": difficulty,
            "relevant_chunks": relevant_chunks,

            "candidate_k": CANDIDATE_K,
            "candidate_count": candidate_count,

            "evidence_depths": list(
                EVIDENCE_DEPTHS
            ),

            "selected_evidence_depth": (
                selected_depth
            ),

            "selected_evidence_count": len(
                selected_evidence
            ),

            "retrieved_chunk_indices": [
                result["chunk_index"]
                for result in retrieved
            ],

            "reranked_chunk_indices": [
                result["chunk_index"]
                for result in reranked
            ],

            "selected_evidence_chunk_indices": [
                result["chunk_index"]
                for result in selected_evidence
            ],

            "relevant_evidence_reached": (
                relevant_reached
            ),

            "evidence_label": final_label,
            "decision": decision,

            "failure_type": failure_type,

            "evidence_rounds": rounds,
        }

        results.append(result)

        print(
            f"[{i:02d}/{total_questions}] "
            f"{question_id} | "
            f"{decision:<8} | "
            f"{final_label:<12} | "
            f"depth={selected_depth:<2} | "
            f"reachable={relevant_reached}"
        )

    # ========================================================
    # METRICS
    # ========================================================

    answerable_support_rate = (
        supported_answerable
        / answerable_count
        if answerable_count
        else 0.0
    )

    unanswerable_abstention_rate = (
        correct_abstentions
        / unanswerable_count
        if unanswerable_count
        else 0.0
    )

    false_support_rate = (
        false_supports
        / unanswerable_count
        if unanswerable_count
        else 0.0
    )

    overall_correct = (
        supported_answerable
        + correct_abstentions
    )

    overall_accuracy = (
        overall_correct
        / total_questions
        if total_questions
        else 0.0
    )

    reachability_rate = (
        retrieval_reached_count
        / answerable_count
        if answerable_count
        else 0.0
    )

    reranking_reachability_rate = (
        reranking_reached_count
        / answerable_count
        if answerable_count
        else 0.0
    )

    average_selected_depth = (
        total_selected_evidence
        / total_questions
        if total_questions
        else 0.0
    )

    average_evidence_rounds = (
        total_evidence_rounds
        / total_questions
        if total_questions
        else 0.0
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "experiment": (
            "fixed_top20_adaptive_evidence"
        ),

        "description": (
            "Fixed candidate depth 20 with "
            "adaptive evidence depth 1, 3, 5, 10"
        ),

        "dataset": {
            "total": total_questions,
            "answerable": answerable_count,
            "unanswerable": unanswerable_count,
        },

        "configuration": {
            "candidate_depth": CANDIDATE_K,
            "evidence_depths": list(
                EVIDENCE_DEPTHS
            ),
            "retrieval_expansion": False,
            "adaptive_evidence": True,
        },

        "outcomes": {
            "supported_answerable": (
                supported_answerable
            ),
            "missed_answerable": (
                missed_answerable
            ),
            "correct_abstentions": (
                correct_abstentions
            ),
            "false_supports": (
                false_supports
            ),
        },

        "rates": {
            "answerable_support_rate": round(
                answerable_support_rate * 100,
                2,
            ),
            "unanswerable_abstention_rate": round(
                unanswerable_abstention_rate * 100,
                2,
            ),
            "false_support_rate": round(
                false_support_rate * 100,
                2,
            ),
            "overall_correct_rate": round(
                overall_accuracy * 100,
                2,
            ),
        },

        "efficiency": {
            "average_selected_evidence_depth": round(
                average_selected_depth,
                3,
            ),
            "average_evidence_rounds": round(
                average_evidence_rounds,
                3,
            ),
        },

        "evidence_depth_distribution": dict(
            sorted(
                evidence_depth_counter.items()
            )
        ),

        "retrieval_reachability": {
            "relevant_evidence_reached_at_candidate_20": (
                retrieval_reached_count
            ),
            "total_answerable": answerable_count,
            "reachability_rate": round(
                reachability_rate * 100,
                2,
            ),
        },

        "reranking_reachability": {
            "relevant_evidence_in_reranked_top10": (
                reranking_reached_count
            ),
            "total_answerable": answerable_count,
            "reachability_rate": round(
                reranking_reachability_rate * 100,
                2,
            ),
        },

        "failure_distribution": dict(
            failure_counter
        ),

        "difficulty": difficulty_stats,
    }

    # ========================================================
    # SAVE
    # ========================================================

    Path(RESULTS_PATH).parent.mkdir(
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

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FIXED TOP-20 + ADAPTIVE EVIDENCE RESULTS")
    print("=" * 70)

    print("\nDataset:")
    print(f"  Total: {total_questions}")
    print(f"  Answerable: {answerable_count}")
    print(f"  Unanswerable: {unanswerable_count}")

    print("\nOutcomes:")
    print(
        f"  Supported answerable: "
        f"{supported_answerable}"
    )
    print(
        f"  Missed answerable: "
        f"{missed_answerable}"
    )
    print(
        f"  Correct abstentions: "
        f"{correct_abstentions}"
    )
    print(
        f"  False supports: "
        f"{false_supports}"
    )

    print("\nRates:")
    print(
        f"  Answerable support rate: "
        f"{answerable_support_rate * 100:.1f}%"
    )
    print(
        f"  Unanswerable abstention rate: "
        f"{unanswerable_abstention_rate * 100:.1f}%"
    )
    print(
        f"  False support rate: "
        f"{false_support_rate * 100:.1f}%"
    )
    print(
        f"  Overall correct rate: "
        f"{overall_accuracy * 100:.1f}%"
    )

    print("\nEfficiency:")
    print(
        f"  Average selected evidence depth: "
        f"{average_selected_depth:.3f}"
    )
    print(
        f"  Average evidence rounds: "
        f"{average_evidence_rounds:.3f}"
    )

    print("\nEvidence depth distribution:")

    for depth, count in sorted(
        evidence_depth_counter.items()
    ):
        print(
            f"  Top-{depth}: {count}"
        )

    print("\nRetrieval reachability:")
    print(
        f"  Relevant evidence reached at Top-20: "
        f"{retrieval_reached_count}/{answerable_count}"
    )
    print(
        f"  Reachability rate: "
        f"{reachability_rate * 100:.1f}%"
    )

    print("\nReranking reachability:")
    print(
        f"  Relevant evidence in reranked Top-10: "
        f"{reranking_reached_count}/{answerable_count}"
    )
    print(
        f"  Reachability rate: "
        f"{reranking_reachability_rate * 100:.1f}%"
    )

    print("\nFailure distribution:")

    for failure_type, count in sorted(
        failure_counter.items()
    ):
        print(
            f"  {failure_type}: {count}"
        )

    print("\nDifficulty:")

    for difficulty, stats in sorted(
        difficulty_stats.items()
    ):
        print(
            f"  {difficulty}: {stats}"
        )

    print("\nSaved:")
    print(f"  {RESULTS_PATH}")
    print(f"  {SUMMARY_PATH}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
