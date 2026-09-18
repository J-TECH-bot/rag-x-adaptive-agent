import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

from evaluation.evidence_sufficiency import check_evidence_sufficiency


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"

TOP_K = 10
CANDIDATE_K = 50
THRESHOLD = 0.40


def main():

    # --------------------------------------------------
    # Load document
    # --------------------------------------------------

    print("Loading PDF...")

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

    print("\nCreating embeddings...")

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

    print("Initializing BM25...")

    bm25_retriever = BM25Retriever(chunks)

    # --------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever
    )

    # --------------------------------------------------
    # Reranker
    # --------------------------------------------------

    print("Loading reranker...")

    reranker = Reranker()

    # --------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------

    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    print(f"Evaluation questions: {len(questions)}")

    # --------------------------------------------------
    # Counters
    # --------------------------------------------------

    answerable_total = 0
    answerable_sufficient = 0
    answerable_insufficient = 0

    unanswerable_total = 0
    unanswerable_correct_abstention = 0
    unanswerable_false_sufficient = 0

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("EVIDENCE SUFFICIENCY EVALUATION")
    print("=" * 80)

    for item in questions:

        question_id = item["id"]
        question = item["question"]
        answerability = item["answerability"]
        reference_answer = item["reference_answer"]

        # ----------------------------------------------
        # Hybrid candidate retrieval
        # ----------------------------------------------

        candidates = hybrid_retriever.retrieve(
            question,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K
        )

        # ----------------------------------------------
        # Cross-encoder reranking
        # ----------------------------------------------

        reranked = reranker.rerank(
            question,
            candidates
        )

        # ----------------------------------------------
        # Take final Top-K evidence
        # ----------------------------------------------

        evidence = reranked[:TOP_K]

        # ----------------------------------------------
        # Evidence sufficiency
        #
        # IMPORTANT:
        # reference_answer is used ONLY for evaluation.
        # It is NOT available to a production system.
        # ----------------------------------------------

        result = check_evidence_sufficiency(
            reference_answer,
            evidence,
            threshold=THRESHOLD
        )

        sufficient = result["sufficient"]
        score = result["score"]

        # ----------------------------------------------
        # Classification
        # ----------------------------------------------

        if answerability == "answerable":

            answerable_total += 1

            if sufficient:
                answerable_sufficient += 1
                classification = "ANSWERABLE + SUFFICIENT"

            else:
                answerable_insufficient += 1
                classification = "ANSWERABLE + INSUFFICIENT"

        else:

            unanswerable_total += 1

            if sufficient:
                unanswerable_false_sufficient += 1
                classification = "UNANSWERABLE + FALSE SUFFICIENT"

            else:
                unanswerable_correct_abstention += 1
                classification = "UNANSWERABLE + CORRECT ABSTENTION"

        # ----------------------------------------------
        # Print result
        # ----------------------------------------------

        print(
            f"\nQ{question_id}: {question}"
        )

        print(
            f"Answerability: {answerability}"
        )

        print(
            f"Evidence score: {score:.3f}"
        )

        print(
            f"Classification: {classification}"
        )

        print(
            f"Top evidence chunks: "
            f"{[r['chunk_index'] for r in evidence]}"
        )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    answerable_sufficiency_rate = (
        answerable_sufficient / answerable_total
        if answerable_total
        else 0.0
    )

    answerable_insufficiency_rate = (
        answerable_insufficient / answerable_total
        if answerable_total
        else 0.0
    )

    correct_abstention_rate = (
        unanswerable_correct_abstention / unanswerable_total
        if unanswerable_total
        else 0.0
    )

    false_sufficient_rate = (
        unanswerable_false_sufficient / unanswerable_total
        if unanswerable_total
        else 0.0
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL EVIDENCE SUFFICIENCY SUMMARY")
    print("=" * 80)

    print(
        f"\nThreshold: {THRESHOLD}"
    )

    print(
        f"Top-K evidence: {TOP_K}"
    )

    print(
        f"Candidate depth: {CANDIDATE_K}"
    )

    print("\nAnswerable questions")
    print("-" * 40)

    print(
        f"Total: {answerable_total}"
    )

    print(
        f"Sufficient evidence: "
        f"{answerable_sufficient} "
        f"({answerable_sufficiency_rate:.3f})"
    )

    print(
        f"Insufficient evidence: "
        f"{answerable_insufficient} "
        f"({answerable_insufficiency_rate:.3f})"
    )

    print("\nUnanswerable questions")
    print("-" * 40)

    print(
        f"Total: {unanswerable_total}"
    )

    print(
        f"Correct abstention: "
        f"{unanswerable_correct_abstention} "
        f"({correct_abstention_rate:.3f})"
    )

    print(
        f"False sufficient: "
        f"{unanswerable_false_sufficient} "
        f"({false_sufficient_rate:.3f})"
    )


if __name__ == "__main__":
    main()