import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

from evaluation.evidence_sufficiency import (
    check_evidence_sufficiency
)

from evaluation.semantic_evidence_sufficiency import (
    semantic_evidence_score
)


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"
OUTPUT_PATH = "data/evaluation/evidence_diagnostics.json"

TOP_K = 10
CANDIDATE_K = 50

# Temporary baseline threshold.
# We are NOT treating this as a calibrated production threshold.
LEXICAL_THRESHOLD = 0.40


def main():

    # ---------------------------------------------------------
    # 1. Load and chunk document
    # ---------------------------------------------------------

    print("Loading PDF...")

    text = load_pdf(PDF_PATH)

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Total chunks: {len(chunks)}")


    # ---------------------------------------------------------
    # 2. Create embeddings
    # ---------------------------------------------------------

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


    # ---------------------------------------------------------
    # 3. Initialize BM25 + Hybrid retrieval
    # ---------------------------------------------------------

    print("Initializing BM25...")

    bm25_retriever = BM25Retriever(chunks)

    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever
    )


    # ---------------------------------------------------------
    # 4. Initialize reranker
    # ---------------------------------------------------------

    print("Loading reranker...")

    reranker = Reranker()


    # ---------------------------------------------------------
    # 5. Load evaluation questions
    # ---------------------------------------------------------

    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    print(f"Evaluation questions: {len(questions)}")


    # ---------------------------------------------------------
    # 6. Evaluate every question
    # ---------------------------------------------------------

    diagnostics = []

    print("\n" + "=" * 80)
    print("CREATING EVIDENCE DIAGNOSTICS")
    print("=" * 80)


    for item in questions:

        question_id = item["id"]
        question = item["question"]
        answerability = item["answerability"]
        reference_answer = item["reference_answer"]


        # -----------------------------------------------------
        # Retrieve candidate pool
        # -----------------------------------------------------

        candidates = hybrid_retriever.retrieve(
            question,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K
        )


        # -----------------------------------------------------
        # Rerank candidates
        # -----------------------------------------------------

        reranked = reranker.rerank(
            question,
            candidates
        )


        # -----------------------------------------------------
        # Final evidence
        # -----------------------------------------------------

        evidence = reranked[:TOP_K]


        # -----------------------------------------------------
        # Lexical evidence score
        #
        # IMPORTANT:
        # reference_answer is GOLD evaluation data.
        # It is NOT production logic.
        # -----------------------------------------------------

        lexical_result = check_evidence_sufficiency(
            reference_answer,
            evidence,
            threshold=LEXICAL_THRESHOLD
        )


        lexical_score = lexical_result["score"]

        lexical_sufficient = lexical_result["sufficient"]


        # -----------------------------------------------------
        # Semantic evidence score
        # -----------------------------------------------------

        semantic_result = semantic_evidence_score(
            question,
            evidence,
            embedder
        )

        semantic_score = semantic_result["score"]


        # -----------------------------------------------------
        # Expected behavior
        #
        # This comes from the evaluation dataset.
        # It is used to evaluate the system, NOT to make
        # the actual retrieval decision.
        # -----------------------------------------------------

        if answerability == "answerable":
            expected_behavior = "answer"
        else:
            expected_behavior = "abstain"


        # -----------------------------------------------------
        # Store diagnostic information
        # -----------------------------------------------------

        diagnostic = {
            "id": question_id,
            "question": question,
            "answerability": answerability,

            "candidate_depth": CANDIDATE_K,
            "top_k": TOP_K,

            "retrieved_chunks": [
                result["chunk_index"]
                for result in evidence
            ],

            "lexical_score": round(
                float(lexical_score),
                4
            ),

            "lexical_sufficient": bool(
                lexical_sufficient
            ),

            "semantic_score": round(
                float(semantic_score),
                4
            ),

            "semantic_sufficient": None,

            "expected_behavior": expected_behavior
        }


        diagnostics.append(diagnostic)


        # -----------------------------------------------------
        # Progress output
        # -----------------------------------------------------

        print(
            f"Q{question_id:02d} | "
            f"{answerability:12s} | "
            f"Lexical: {lexical_score:.3f} | "
            f"Semantic: {semantic_score:.3f}"
        )


    # ---------------------------------------------------------
    # 7. Save diagnostics
    # ---------------------------------------------------------

    with open(OUTPUT_PATH, "w") as f:

        json.dump(
            diagnostics,
            f,
            indent=2
        )


    # ---------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------

    answerable = [
        x for x in diagnostics
        if x["answerability"] == "answerable"
    ]

    unanswerable = [
        x for x in diagnostics
        if x["answerability"] == "unanswerable"
    ]


    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    print(f"Total questions: {len(diagnostics)}")
    print(f"Answerable: {len(answerable)}")
    print(f"Unanswerable: {len(unanswerable)}")

    print(f"\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()