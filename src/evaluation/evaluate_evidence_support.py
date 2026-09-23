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

QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"
PDF_PATH = "data/raw/rag_original.pdf"

TOP_K = 10
CANDIDATE_K = 50

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions(path: str):
    """
    Load evaluation questions from JSON.
    """

    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# LOAD AND CHUNK PDF
# ============================================================

def load_chunks():
    """
    Load PDF text using the project's canonical PDF loader
    and split it using the project's canonical sentence-aware
    chunker.
    """

    text = load_pdf(PDF_PATH)

    chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return chunks


# ============================================================
# COMBINE RETRIEVED EVIDENCE
# ============================================================

def combine_evidence(results: list[dict]) -> str:
    """
    Combine retrieved chunks into one evidence string.
    """

    return "\n\n".join(
        result["text"]
        for result in results
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 80)
    print("EVIDENCE SUPPORT EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load evaluation questions
    # --------------------------------------------------------

    print("\nLoading questions...")

    questions = load_questions(
        QUESTIONS_PATH
    )

    print(
        f"Questions loaded: {len(questions)}"
    )

    # --------------------------------------------------------
    # 2. Load document chunks
    # --------------------------------------------------------

    print("\nLoading document chunks...")

    chunks = load_chunks()

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    # --------------------------------------------------------
    # 3. Load embedding model
    # --------------------------------------------------------

    print("\nLoading retrieval components...")

    embedder = Embedder()

    # --------------------------------------------------------
    # 4. Create FAISS vector store
    # --------------------------------------------------------

    vector_store = VectorStore(
        dimension=384
    )

    # Embed all document chunks
    chunk_embeddings = embedder.embed(
        chunks
    )

    # IMPORTANT:
    # VectorStore uses .add(), not .build()
    vector_store.add(
        chunk_embeddings
    )

    # --------------------------------------------------------
    # 5. Dense retriever
    # --------------------------------------------------------

    dense_retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks
    )

    # --------------------------------------------------------
    # 6. BM25 retriever
    # --------------------------------------------------------

    bm25_retriever = BM25Retriever(
        chunks
    )

    # --------------------------------------------------------
    # 7. Hybrid retriever
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever
    )

    # --------------------------------------------------------
    # 8. Cross-encoder reranker
    # --------------------------------------------------------

    reranker = Reranker()

    # --------------------------------------------------------
    # 9. Evidence Support Judge
    # --------------------------------------------------------

    print("\nLoading Evidence Support Judge...")

    judge = EvidenceSupportJudge()

    # --------------------------------------------------------
    # 10. Evaluate all questions
    # --------------------------------------------------------

    print("\nStarting evaluation...\n")

    results = []

    for i, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        # Gold answerability is used ONLY to evaluate
        # the judge's behavior.
        #
        # It is NOT passed to the retrieval system
        # or Evidence Support Judge.

        expected_behavior = (
            "answer"
            if item["answerability"] == "answerable"
            else "abstain"
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        candidates = hybrid_retriever.retrieve(
            question,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K
        )

        # ----------------------------------------------------
        # Reranking
        # ----------------------------------------------------

        reranked = reranker.rerank(
            question,
            candidates
        )

        # ----------------------------------------------------
        # Select final evidence
        # ----------------------------------------------------

        evidence = reranked[:TOP_K]

        evidence_text = combine_evidence(
            evidence
        )

        # ----------------------------------------------------
        # Evidence Support Judge
        # ----------------------------------------------------

        judgment = judge.judge(
            question,
            evidence_text
        )

        predicted_label = judgment["label"]

        predicted_behavior = (
            "answer"
            if predicted_label == "SUPPORTED"
            else "abstain"
        )

        correct_behavior = (
            predicted_behavior == expected_behavior
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "id": item["id"],
            "question": question,
            "answerability": item["answerability"],
            "difficulty_type": item["difficulty_type"],
            "expected_behavior": expected_behavior,
            "prediction": predicted_label,
            "predicted_behavior": predicted_behavior,
            "correct_behavior": correct_behavior,
            "raw_output": judgment["raw_output"]
        }

        results.append(result)

        status = "✓" if correct_behavior else "✗"

        print(
            f"[{i:02d}/{len(questions)}] "
            f"Q{item['id']} | "
            f"Expected: {expected_behavior:<7} | "
            f"Predicted: {predicted_behavior:<7} | "
            f"{status}"
        )

    # ========================================================
    # ANALYSIS
    # ========================================================

    answerable = [
        r
        for r in results
        if r["answerability"] == "answerable"
    ]

    unanswerable = [
        r
        for r in results
        if r["answerability"] == "unanswerable"
    ]

    # --------------------------------------------------------
    # Answerable
    # --------------------------------------------------------

    answerable_supported = sum(
        r["prediction"] == "SUPPORTED"
        for r in answerable
    )

    answerable_insufficient = sum(
        r["prediction"] == "INSUFFICIENT"
        for r in answerable
    )

    # --------------------------------------------------------
    # Unanswerable
    # --------------------------------------------------------

    unanswerable_supported = sum(
        r["prediction"] == "SUPPORTED"
        for r in unanswerable
    )

    unanswerable_insufficient = sum(
        r["prediction"] == "INSUFFICIENT"
        for r in unanswerable
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    correct = sum(
        r["correct_behavior"]
        for r in results
    )

    accuracy = (
        correct / len(results)
        if results
        else 0.0
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"\nTotal questions: {len(results)}"
    )

    print("\nANSWERABLE")
    print("-" * 40)

    print(
        f"SUPPORTED:    "
        f"{answerable_supported}/{len(answerable)}"
    )

    print(
        f"INSUFFICIENT: "
        f"{answerable_insufficient}/{len(answerable)}"
    )

    print("\nUNANSWERABLE")
    print("-" * 40)

    print(
        f"SUPPORTED (false positive): "
        f"{unanswerable_supported}/{len(unanswerable)}"
    )

    print(
        f"INSUFFICIENT (correct abstention): "
        f"{unanswerable_insufficient}/{len(unanswerable)}"
    )

    print("\nOVERALL")
    print("-" * 40)

    print(
        f"Correct behavior: "
        f"{correct}/{len(results)}"
    )

    print(
        f"Behavior accuracy: "
        f"{accuracy:.3f}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_path = (
        "data/evaluation/"
        "evidence_support_results.json"
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        f"\nSaved results to: "
        f"{output_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
