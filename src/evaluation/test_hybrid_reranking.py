import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from evaluation.evaluator import recall_at_k, reciprocal_rank


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"


def main():

    # --------------------------------------------------
    # 1. Load PDF and create baseline chunks
    # --------------------------------------------------

    text = load_pdf(PDF_PATH)

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Total chunks: {len(chunks)}")

    # --------------------------------------------------
    # 2. Build Dense Retriever
    # --------------------------------------------------

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
    # 3. Build BM25 Retriever
    # --------------------------------------------------

    bm25_retriever = BM25Retriever(chunks)

    # --------------------------------------------------
    # 4. Build Hybrid Retriever
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever
    )

    # --------------------------------------------------
    # 5. Build Reranker
    # --------------------------------------------------

    reranker = Reranker()

    # --------------------------------------------------
    # 6. Load evaluation questions
    # --------------------------------------------------

    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    recall_1 = []
    recall_3 = []
    recall_5 = []
    mrr_scores = []

    print("\n" + "=" * 70)
    print("HYBRID + RERANKING EVALUATION")
    print("=" * 70)

    # --------------------------------------------------
    # 7. Evaluate
    # --------------------------------------------------

    for item in questions:

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        # Hybrid retrieves 20 candidates
        hybrid_results = hybrid_retriever.retrieve(
            question,
            top_k=20,
            candidate_k=20
        )

        # Cross-encoder reranks the 20 candidates
        reranked_results = reranker.rerank(
            question,
            hybrid_results
        )

        # Evaluate final Top-5
        final_results = reranked_results[:5]

        r1 = recall_at_k(
            final_results,
            relevant_chunks,
            1
        )

        r3 = recall_at_k(
            final_results,
            relevant_chunks,
            3
        )

        r5 = recall_at_k(
            final_results,
            relevant_chunks,
            5
        )

        mrr = reciprocal_rank(
            final_results,
            relevant_chunks
        )

        recall_1.append(r1)
        recall_3.append(r3)
        recall_5.append(r5)
        mrr_scores.append(mrr)

        print(f"\nQuestion: {question}")
        print(f"Relevant chunks: {relevant_chunks}")

        print(
            "Hybrid candidates:",
            [result["chunk_index"] for result in hybrid_results]
        )

        print(
            "Reranked Top-5:",
            [result["chunk_index"] for result in final_results]
        )

        print(f"Recall@1: {r1}")
        print(f"Recall@3: {r3}")
        print(f"Recall@5: {r5}")
        print(f"MRR: {mrr:.3f}")

    # --------------------------------------------------
    # 8. Summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID + RERANKING SUMMARY")
    print("=" * 70)

    print(
        f"Recall@1: {sum(recall_1) / len(recall_1):.3f}"
    )

    print(
        f"Recall@3: {sum(recall_3) / len(recall_3):.3f}"
    )

    print(
        f"Recall@5: {sum(recall_5) / len(recall_5):.3f}"
    )

    print(
        f"MRR:      {sum(mrr_scores) / len(mrr_scores):.3f}"
    )


if __name__ == "__main__":
    main()