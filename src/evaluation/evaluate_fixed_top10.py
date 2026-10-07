from __future__ import annotations

import json
from collections import Counter, defaultdict
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


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = PROJECT_ROOT / "data" / "raw" / "rag_original.pdf"
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_questions.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "fixed_top10_results.json"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "fixed_top10_summary.json"
)

FIXED_TOP_K = 10


# ============================================================
# DATASET
# ============================================================

def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("retrieval_questions.json must contain a list.")

    return questions


# ============================================================
# BUILD RETRIEVAL SYSTEM
# ============================================================

def build_system(chunks: list[str]):
    print("=" * 80)
    print("BUILDING FIXED TOP-10 BASELINE")
    print("=" * 80)

    print("\nLoading embedding model...")
    embedder = Embedder("all-MiniLM-L6-v2")

    print("Embedding document chunks...")
    embeddings = embedder.embed(chunks)

    print(f"Embedding shape: {embeddings.shape}")

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(embeddings)

    dense_retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks,
    )

    bm25_retriever = BM25Retriever(chunks)

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    print("\nLoading reranker...")
    reranker = Reranker()

    print("\nLoading evidence judge...")
    evidence_judge = EvidenceSupportJudgeV2()

    print("\nFixed Top-10 baseline ready.")

    return hybrid_retriever, reranker, evidence_judge


# ============================================================
# EVIDENCE
# ============================================================

def combine_evidence(results: list[dict]) -> str:
    return "\n\n".join(
        result["text"]
        for result in results
    )


# ============================================================
# QUESTION EVALUATION
# ============================================================

def evaluate_question(
    question_item: dict,
    hybrid_retriever: HybridRetriever,
    reranker: Reranker,
    evidence_judge: EvidenceSupportJudgeV2,
) -> dict:

    question_id = question_item["id"]
    question = question_item["question"]
    answerability = question_item["answerability"]
    difficulty = question_item["difficulty_type"]

    relevant_chunks = set(
        question_item.get("relevant_chunks", [])
    )

    # --------------------------------------------------------
    # Fixed Top-10 hybrid retrieval
    # --------------------------------------------------------

    retrieved = hybrid_retriever.retrieve(
        question,
        top_k=FIXED_TOP_K,
        candidate_k=FIXED_TOP_K,
    )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    reranked = reranker.rerank(
        question,
        retrieved,
    )

    evidence = reranked[:FIXED_TOP_K]

    evidence_chunk_indices = [
        result["chunk_index"]
        for result in evidence
    ]

    # --------------------------------------------------------
    # Dataset relevance diagnostic
    # --------------------------------------------------------

    relevant_reached = bool(
        relevant_chunks.intersection(
            evidence_chunk_indices
        )
    )

    # --------------------------------------------------------
    # Evidence judge
    # --------------------------------------------------------

    evidence_text = combine_evidence(evidence)

    judgment = evidence_judge.judge(
        question,
        evidence_text,
    )

    evidence_label = (
        judgment["label"]
        .strip()
        .upper()
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if evidence_label == "SUPPORTED":
        decision = "ANSWER"
    else:
        decision = "ABSTAIN"

    # --------------------------------------------------------
    # Outcome
    # --------------------------------------------------------

    if answerability == "answerable":

        if (
            decision == "ANSWER"
            and relevant_reached
        ):
            outcome = "SUPPORTED_ANSWERABLE"
            failure_diagnosis = "none"

        elif not relevant_reached:
            outcome = "MISSED_ANSWERABLE"
            failure_diagnosis = "retrieval_or_ranking_failure"

        else:
            outcome = "MISSED_ANSWERABLE"
            failure_diagnosis = "evidence_support_failure"

    else:

        if decision == "ABSTAIN":
            outcome = "CORRECT_ABSTENTION"
            failure_diagnosis = "none"
        else:
            outcome = "FALSE_SUPPORT"
            failure_diagnosis = "unsupported_acceptance"

    return {
        "id": question_id,
        "question": question,
        "answerability": answerability,
        "difficulty_type": difficulty,
        "relevant_chunks": list(relevant_chunks),

        "decision": decision,
        "outcome": outcome,
        "failure_diagnosis": failure_diagnosis,

        "candidate_depth": FIXED_TOP_K,
        "candidate_count": len(retrieved),

        "retrieval_rounds": 1,
        "retrieval_expanded": False,

        "evidence_depth": len(evidence),
        "evidence_label": evidence_label,

        "relevant_evidence_reached": relevant_reached,

        "evidence_chunk_indices": evidence_chunk_indices,

        "retrieved_results": retrieved,
        "reranked_results": reranked,

        "judge_raw_output": judgment.get(
            "raw_output",
            ""
        ),

        "reason": (
            "Fixed Top-10 baseline answered because "
            "the evidence judge returned SUPPORTED."
            if decision == "ANSWER"
            else
            "Fixed Top-10 baseline abstained because "
            "the evidence judge did not return SUPPORTED."
        ),
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(results: list[dict]) -> dict:

    total = len(results)

    answerable = [
        r for r in results
        if r["answerability"] == "answerable"
    ]

    unanswerable = [
        r for r in results
        if r["answerability"] == "unanswerable"
    ]

    supported_answerable = [
        r for r in results
        if r["outcome"] == "SUPPORTED_ANSWERABLE"
    ]

    missed_answerable = [
        r for r in results
        if r["outcome"] == "MISSED_ANSWERABLE"
    ]

    correct_abstentions = [
        r for r in results
        if r["outcome"] == "CORRECT_ABSTENTION"
    ]

    false_supports = [
        r for r in results
        if r["outcome"] == "FALSE_SUPPORT"
    ]

    relevant_reached = [
        r for r in answerable
        if r["relevant_evidence_reached"]
    ]

    failure_distribution = Counter(
        r["failure_diagnosis"]
        for r in results
    )

    difficulty_summary = defaultdict(
        lambda: {
            "total": 0,
            "supported": 0,
            "missed": 0,
            "correct_abstention": 0,
            "false_support": 0,
        }
    )

    for result in results:

        difficulty = result["difficulty_type"]

        difficulty_summary[difficulty]["total"] += 1

        if result["outcome"] == "SUPPORTED_ANSWERABLE":
            difficulty_summary[difficulty]["supported"] += 1

        elif result["outcome"] == "MISSED_ANSWERABLE":
            difficulty_summary[difficulty]["missed"] += 1

        elif result["outcome"] == "CORRECT_ABSTENTION":
            difficulty_summary[difficulty][
                "correct_abstention"
            ] += 1

        elif result["outcome"] == "FALSE_SUPPORT":
            difficulty_summary[difficulty][
                "false_support"
            ] += 1

    for difficulty, values in difficulty_summary.items():

        difficulty_total = values["total"]

        if difficulty_total > 0:
            values["accuracy"] = round(
                (
                    values["supported"]
                    + values["correct_abstention"]
                )
                / difficulty_total,
                4,
            )
        else:
            values["accuracy"] = 0.0

    overall_correct = (
        len(supported_answerable)
        + len(correct_abstentions)
    )

    summary = {
        "experiment": "fixed_top10_baseline",

        "configuration": {
            "retrieval": "HybridRetriever",
            "candidate_depth": FIXED_TOP_K,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "evidence_depth": FIXED_TOP_K,
            "evidence_judge": (
                "evaluation.evidence_support_v2."
                "EvidenceSupportJudgeV2"
            ),
            "evidence_judge_model": "google/flan-t5-base",
            "adaptive_retrieval": False,
            "adaptive_evidence_selection": False,
        },

        "dataset": {
            "total": total,
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
        },

        "outcomes": {
            "supported_answerable": len(
                supported_answerable
            ),
            "missed_answerable": len(
                missed_answerable
            ),
            "correct_abstentions": len(
                correct_abstentions
            ),
            "false_supports": len(
                false_supports
            ),
        },

        "rates": {
            "answerable_support_rate": round(
                len(supported_answerable)
                / len(answerable),
                4,
            ),
            "unanswerable_abstention_rate": round(
                len(correct_abstentions)
                / len(unanswerable),
                4,
            ),
            "false_support_rate": round(
                len(false_supports)
                / len(unanswerable),
                4,
            ),
            "overall_correct_rate": round(
                overall_correct / total,
                4,
            ),
        },

        "efficiency": {
            "fixed_candidate_depth": FIXED_TOP_K,
            "fixed_evidence_depth": FIXED_TOP_K,
            "average_candidate_depth": FIXED_TOP_K,
            "average_evidence_depth": FIXED_TOP_K,
            "retrieval_rounds": 1,
            "retrieval_expansions": 0,
        },

        "retrieval_reachability": {
            "answerable_relevant_evidence_reached": len(
                relevant_reached
            ),
            "answerable_total": len(answerable),
            "rate": round(
                len(relevant_reached)
                / len(answerable),
                4,
            ),
        },

        "failure_distribution": dict(
            failure_distribution
        ),

        "difficulty_summary": dict(
            difficulty_summary
        ),
    }

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RAG-X FIXED TOP-10 BASELINE — 60 QUESTION EVALUATION")
    print("=" * 80)

    questions = load_questions()

    print(f"\nQuestions: {len(questions)}")

    answerability_counts = Counter(
        item["answerability"]
        for item in questions
    )

    difficulty_counts = Counter(
        item["difficulty_type"]
        for item in questions
    )

    print("\nAnswerability:")

    for key, value in answerability_counts.items():
        print(f"  {key}: {value}")

    print("\nDifficulty:")

    for key, value in difficulty_counts.items():
        print(f"  {key}: {value}")

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    print("\nLoading PDF...")

    text = load_pdf(str(PDF_PATH))

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"Chunks: {len(chunks)}")

    # --------------------------------------------------------
    # Build system
    # --------------------------------------------------------

    (
        hybrid_retriever,
        reranker,
        evidence_judge,
    ) = build_system(chunks)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RUNNING FIXED TOP-10 EVALUATION")
    print("=" * 80)

    results = []

    for position, question_item in enumerate(
        questions,
        start=1,
    ):

        question_id = question_item["id"]
        question = question_item["question"]

        print("\n" + "=" * 80)
        print(
            f"[{position}/{len(questions)}] "
            f"Q{question_id}: {question}"
        )

        print(
            f"Expected: "
            f"{question_item['answerability']} | "
            f"Difficulty: "
            f"{question_item['difficulty_type']}"
        )

        result = evaluate_question(
            question_item=question_item,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            evidence_judge=evidence_judge,
        )

        results.append(result)

        print(
            f"Decision: {result['decision']}"
        )

        print(
            f"Outcome: {result['outcome']}"
        )

        print(
            f"Candidate depth: "
            f"{result['candidate_depth']}"
        )

        print(
            f"Candidate count: "
            f"{result['candidate_count']}"
        )

        print(
            f"Evidence depth: "
            f"{result['evidence_depth']}"
        )

        print(
            f"Relevant evidence reached: "
            f"{result['relevant_evidence_reached']}"
        )

        print(
            f"Evidence label: "
            f"{result['evidence_label']}"
        )

        print(
            f"Failure diagnosis: "
            f"{result['failure_diagnosis']}"
        )

        print(
            f"Reason: {result['reason']}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(results)

    print("\n" + "=" * 80)
    print("FINAL FIXED TOP-10 BASELINE RESULTS")
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
        f"{summary['rates']['answerable_support_rate'] * 100:.1f}%"
    )

    print(
        f"  Unanswerable abstention rate: "
        f"{summary['rates']['unanswerable_abstention_rate'] * 100:.1f}%"
    )

    print(
        f"  False support rate: "
        f"{summary['rates']['false_support_rate'] * 100:.1f}%"
    )

    print(
        f"  Overall correct rate: "
        f"{summary['rates']['overall_correct_rate'] * 100:.1f}%"
    )

    print("\nEfficiency:")

    print(
        f"  Fixed candidate depth: "
        f"{FIXED_TOP_K}"
    )

    print(
        f"  Fixed evidence depth: "
        f"{FIXED_TOP_K}"
    )

    print(
        "  Retrieval expansions: 0"
    )

    print("\nRetrieval reachability:")

    print(
        f"  Relevant evidence reached at Top-10: "
        f"{summary['retrieval_reachability']['answerable_relevant_evidence_reached']}/"
        f"{summary['retrieval_reachability']['answerable_total']}"
    )

    print(
        f"  Reachability rate: "
        f"{summary['retrieval_reachability']['rate'] * 100:.1f}%"
    )

    print("\nFailure distribution:")

    for key, value in summary[
        "failure_distribution"
    ].items():
        print(f"  {key}: {value}")

    print("\nDifficulty summary:")

    for difficulty, values in sorted(
        summary["difficulty_summary"].items()
    ):

        print(f"\n  {difficulty}")

        print(
            f"    Total: "
            f"{values['total']}"
        )

        print(
            f"    Supported: "
            f"{values['supported']}"
        )

        print(
            f"    Missed: "
            f"{values['missed']}"
        )

        print(
            f"    Correct abstention: "
            f"{values['correct_abstention']}"
        )

        print(
            f"    False support: "
            f"{values['false_support']}"
        )

        print(
            f"    Accuracy: "
            f"{values['accuracy'] * 100:.1f}%"
        )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(f"\nDetailed results:\n{RESULTS_PATH}")
    print(f"\nSummary:\n{SUMMARY_PATH}")

    print("\nBaseline evaluation complete.")


if __name__ == "__main__":
    main()
