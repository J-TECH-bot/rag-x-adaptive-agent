from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text

from embeddings.embedder import Embedder

from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

from evaluation.evidence_support_v2 import EvidenceSupportJudgeV2

from adaptive.decision_engine import (
    AdaptiveDecisionEngine,
)

from adaptive.adaptive_retriever import (
    AdaptiveRetriever,
)


PDF_PATH = "data/raw/rag_original.pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def build_system():

    print("\nLoading document...")

    text = load_pdf(PDF_PATH)

    chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    print(f"Chunks: {len(chunks)}")

    print("\nLoading embedding model...")

    embedder = Embedder()

    print("Building FAISS index...")

    vector_store = VectorStore(
        dimension=384
    )

    embeddings = embedder.embed(
        chunks
    )

    vector_store.add(
        embeddings
    )

    dense_retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks
    )

    bm25_retriever = BM25Retriever(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever
    )

    print("\nLoading reranker...")

    reranker = Reranker()

    print("\nLoading evidence judge...")

    evidence_judge = EvidenceSupportJudgeV2()

    decision_engine = AdaptiveDecisionEngine(
        max_retrieval_rounds=2
    )

    adaptive_retriever = AdaptiveRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        evidence_judge=evidence_judge,
        decision_engine=decision_engine,
        evidence_top_k=10,
        initial_candidate_k=20,
        expanded_candidate_k=50
    )

    return adaptive_retriever


def print_result(result):

    print("\n" + "=" * 80)
    print("ADAPTIVE RAG RESULT")
    print("=" * 80)

    print(
        f"\nQuery:\n{result['query']}"
    )

    print(
        f"\nFinal decision: "
        f"{result['final_decision']}"
    )

    print(
        f"Final evidence label: "
        f"{result['final_evidence_label']}"
    )

    print(
        f"Rounds used: "
        f"{result['rounds_used']}"
    )

    print(
        f"Reason:\n"
        f"{result['final_reason']}"
    )

    print("\nTRACE")
    print("-" * 80)

    for round_result in result["trace"]:

        print(
            f"Round {round_result['round']} | "
            f"candidate_k={round_result['candidate_k']} | "
            f"candidates={round_result['candidate_count']} | "
            f"evidence={round_result['evidence_count']} | "
            f"label={round_result['evidence_label']} | "
            f"decision={round_result['decision']}"
        )

        print(
            f"Reason: {round_result['reason']}"
        )

    print("\nTOP EVIDENCE")
    print("-" * 80)

    for i, item in enumerate(
        result["evidence"][:3],
        start=1
    ):

        print(
            f"\nEvidence {i} "
            f"(chunk={item['chunk_index']})"
        )

        print(
            item["text"][:500]
        )


def main():

    adaptive_retriever = build_system()

    test_queries = [
        (
            "What is Retrieval-Augmented Generation?"
        ),
        (
            "What are the main components of the RAG framework?"
        ),
        (
            "What is the capital of France?"
        )
    ]

    for query in test_queries:

        result = adaptive_retriever.retrieve(
            query
        )

        print_result(result)


if __name__ == "__main__":
    main()
