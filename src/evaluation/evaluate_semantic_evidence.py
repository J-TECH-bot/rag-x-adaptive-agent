import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

from evaluation.semantic_evidence_sufficiency import (
    semantic_evidence_score
)


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"

TOP_K = 10
CANDIDATE_K = 50


def main():

    # --------------------------------------------------
    # Load PDF
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
    # Questions
    # --------------------------------------------------

    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    print(f"Evaluation questions: {len(questions)}")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("SEMANTIC EVIDENCE EVALUATION")
    print("=" * 80)

    for item in questions:

        question_id = item["id"]
        question = item["question"]
        answerability = item["answerability"]

        # ----------------------------------------------
        # Retrieve candidates
        # ----------------------------------------------

        candidates = hybrid_retriever.retrieve(
            question,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K
        )

        # ----------------------------------------------
        # Rerank candidates
        # ----------------------------------------------

        reranked = reranker.rerank(
            question,
            candidates
        )

        # ----------------------------------------------
        # Final evidence
        # ----------------------------------------------

        evidence = reranked[:TOP_K]

        # ----------------------------------------------
        # Semantic score
        # ----------------------------------------------

        result = semantic_evidence_score(
            question,
            evidence,
            embedder
        )

        score = result["score"]

        # ----------------------------------------------
        # Print
        # ----------------------------------------------

        print(
            f"\nQ{question_id}: {question}"
        )

        print(
            f"Answerability: {answerability}"
        )

        print(
            f"Maximum semantic similarity: {score:.3f}"
        )

        print(
            "Top semantic chunks: "
            f"{result['chunk_scores'][:5]}"
        )


if __name__ == "__main__":
    main()