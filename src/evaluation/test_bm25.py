import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from retrieval.bm25_retriever import BM25Retriever
from evaluation.evaluator import recall_at_k, reciprocal_rank


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"


def main():

    # Load PDF
    text = load_pdf(PDF_PATH)

    # Create the same chunks used in our current baseline
    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Total chunks: {len(chunks)}")

    # Build BM25 retriever
    retriever = BM25Retriever(chunks)

    # Load evaluation questions
    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    recall_1 = []
    recall_3 = []
    recall_5 = []
    mrr_scores = []

    print("\n" + "=" * 70)
    print("BM25 RETRIEVAL EVALUATION")
    print("=" * 70)

    for item in questions:

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        results = retriever.retrieve(
            question,
            top_k=5
        )

        r1 = recall_at_k(
            results,
            relevant_chunks,
            1
        )

        r3 = recall_at_k(
            results,
            relevant_chunks,
            3
        )

        r5 = recall_at_k(
            results,
            relevant_chunks,
            5
        )

        mrr = reciprocal_rank(
            results,
            relevant_chunks
        )

        recall_1.append(r1)
        recall_3.append(r3)
        recall_5.append(r5)
        mrr_scores.append(mrr)

        print(f"\nQuestion: {question}")
        print(f"Relevant chunks: {relevant_chunks}")

        print(
            "Retrieved chunks:",
            [result["chunk_index"] for result in results]
        )

        print(f"Recall@1: {r1}")
        print(f"Recall@3: {r3}")
        print(f"Recall@5: {r5}")
        print(f"MRR: {mrr:.3f}")

    print("\n" + "=" * 70)
    print("BM25 SUMMARY")
    print("=" * 70)

    print(f"Recall@1: {sum(recall_1) / len(recall_1):.3f}")
    print(f"Recall@3: {sum(recall_3) / len(recall_3):.3f}")
    print(f"Recall@5: {sum(recall_5) / len(recall_5):.3f}")
    print(f"MRR:      {sum(mrr_scores) / len(mrr_scores):.3f}")


if __name__ == "__main__":
    main()