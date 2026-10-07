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


def build_system():
    print("\n" + "=" * 80)
    print("BUILDING RAG-X ADAPTIVE RETRIEVAL SYSTEM")
    print("=" * 80)

    # ==============================================================
    # 1. LOAD PDF
    # ==============================================================
    print("\nLoading document...")

    text = load_pdf(str(PDF_PATH))

    if not text.strip():
        raise ValueError("PDF extraction returned empty text.")

    # ==============================================================
    # 2. CHUNK DOCUMENT
    # ==============================================================
    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"Chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    # ==============================================================
    # 3. EMBEDDING MODEL
    # ==============================================================
    print("\nLoading embedding model...")

    embedder = Embedder(
        model_name="all-MiniLM-L6-v2"
    )

    embeddings = embedder.embed(chunks)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # ==============================================================
    # 4. BUILD FAISS VECTOR STORE
    # ==============================================================
    print("\nBuilding FAISS index...")

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(embeddings)

    # ==============================================================
    # 5. DENSE RETRIEVER
    # ==============================================================
    dense_retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks,
    )

    # ==============================================================
    # 6. BM25 RETRIEVER
    # ==============================================================
    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    # ==============================================================
    # 7. HYBRID RETRIEVER
    # ==============================================================
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    # ==============================================================
    # 8. RERANKER
    # ==============================================================
    print("\nLoading reranker...")

    reranker = Reranker()

    # ==============================================================
    # 9. EVIDENCE JUDGE
    # ==============================================================
    print("\nLoading evidence judge...")

    evidence_judge = EvidenceSupportJudgeV2()

    # ==============================================================
    # 10. DECISION ENGINE
    # ==============================================================
    decision_engine = AdaptiveDecisionEngine(
        max_retrieval_rounds=2
    )

    # ==============================================================
    # 11. ADAPTIVE EVIDENCE SELECTOR
    # ==============================================================
    evidence_selector = AdaptiveEvidenceSelector(
        evidence_judge=evidence_judge,
        evidence_depths=(1, 3, 5, 10),
    )

    # ==============================================================
    # 12. ADAPTIVE RETRIEVER
    # ==============================================================
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


def print_result(result: dict):
    print("\n" + "=" * 80)
    print("ADAPTIVE RAG RESULT")
    print("=" * 80)

    print("\nQuery:")
    print(result["query"])

    print(
        f"\nFinal decision: "
        f"{result['decision']}"
    )

    print(
        f"Final evidence label: "
        f"{result['evidence_label']}"
    )

    print(
        f"Selected evidence depth: "
        f"{result['selected_evidence_depth']}"
    )

    print("\nReason:")
    print(result["reason"])

    # ==============================================================
    # TRACE
    # ==============================================================
    print("\nTRACE")
    print("-" * 80)

    for trace in result["trace"]:

        print(
            f"\nRetrieval Round "
            f"{trace['retrieval_round']}"
        )

        print(
            f"Candidate depth: "
            f"{trace['candidate_k']}"
        )

        print(
            f"Candidates retrieved: "
            f"{trace['candidate_count']}"
        )

        print(
            f"Decision: "
            f"{trace['decision']}"
        )

        print(
            f"Evidence label: "
            f"{trace['evidence_label']}"
        )

        print(
            f"Selected evidence depth: "
            f"{trace['selected_evidence_depth']}"
        )

        print("\nEvidence escalation:")

        for evidence_round in trace["evidence_rounds"]:

            print(
                f"  Top-{evidence_round['depth']} | "
                f"evidence="
                f"{evidence_round['evidence_count']} | "
                f"label="
                f"{evidence_round['evidence_label']}"
            )

    # ==============================================================
    # EVIDENCE
    # ==============================================================
    print("\nTOP EVIDENCE")
    print("-" * 80)

    for i, evidence in enumerate(
        result["evidence"],
        start=1,
    ):

        print(
            f"\nEvidence {i} "
            f"(chunk="
            f"{evidence.get('chunk_index', 'N/A')})"
        )

        print(
            f"Score: "
            f"{evidence.get('score', 'N/A')}"
        )

        print(
            evidence["text"][:1200]
        )


def run_query(
    adaptive_retriever,
    query: str,
):
    print("\n\n" + "#" * 80)
    print("RUNNING QUERY:")
    print(query)
    print("#" * 80)

    result = adaptive_retriever.retrieve(
        query
    )

    print_result(result)

    return result


def validate_common_result(
    result: dict,
):
    assert result["decision"] in {
        "ANSWER",
        "ABSTAIN",
    }

    assert result["evidence_label"] in {
        "SUPPORTED",
        "INSUFFICIENT",
        "CONTRADICTED",
        "UNKNOWN",
    }

    assert result["trace"], (
        "Adaptive trace must not be empty."
    )

    assert (
        result["selected_evidence_depth"] >= 1
    )

    for trace in result["trace"]:

        assert trace["decision"] in {
            "ANSWER",
            "EXPAND_RETRIEVAL",
            "ABSTAIN",
        }

        assert trace["candidate_k"] in {
            20,
            50,
        }

        assert trace["candidate_count"] > 0

        assert trace["evidence_rounds"]


def main():

    adaptive_retriever = build_system()

    # ==============================================================
    # TEST 1
    # ==============================================================
    result_1 = run_query(
        adaptive_retriever,
        "What is Retrieval-Augmented Generation?",
    )

    validate_common_result(result_1)

    assert result_1["decision"] == "ANSWER"

    assert (
        result_1["evidence_label"]
        == "SUPPORTED"
    )

    assert len(result_1["trace"]) == 1

    assert (
        result_1["trace"][0]["candidate_k"]
        == 20
    )

    # ==============================================================
    # TEST 2
    # ==============================================================
    result_2 = run_query(
        adaptive_retriever,
        "What are the main components of the RAG framework?",
    )

    validate_common_result(result_2)

    assert result_2["decision"] == "ANSWER"

    assert (
        result_2["evidence_label"]
        == "SUPPORTED"
    )

    assert len(result_2["trace"]) == 1

    assert (
        result_2["trace"][0]["candidate_k"]
        == 20
    )

    # ==============================================================
    # TEST 3
    #
    # Out-of-corpus query.
    #
    # Expected:
    #
    # Round 1:
    #   candidate 20
    #   Top-1
    #   Top-3
    #   Top-5
    #   Top-10
    #   insufficient
    #   EXPAND_RETRIEVAL
    #
    # Round 2:
    #   candidate 50
    #   Top-1
    #   Top-3
    #   Top-5
    #   Top-10
    #   insufficient
    #   ABSTAIN
    # ==============================================================
    result_3 = run_query(
        adaptive_retriever,
        "What is the capital of France?",
    )

    validate_common_result(result_3)

    assert result_3["decision"] == "ABSTAIN"

    assert len(result_3["trace"]) == 2

    # First retrieval round must expand.
    assert (
        result_3["trace"][0]["candidate_k"]
        == 20
    )

    assert (
        result_3["trace"][0]["decision"]
        == "EXPAND_RETRIEVAL"
    )

    # Second retrieval round must abstain.
    assert (
        result_3["trace"][1]["candidate_k"]
        == 50
    )

    assert (
        result_3["trace"][1]["decision"]
        == "ABSTAIN"
    )

    # Both retrieval rounds should exhaust
    # the complete evidence-depth schedule.
    for trace in result_3["trace"]:

        depths = [
            item["depth"]
            for item in trace["evidence_rounds"]
        ]

        assert depths == [
            1,
            3,
            5,
            10,
        ]

    # ==============================================================
    # SUCCESS
    # ==============================================================
    print("\n" + "=" * 80)
    print("ALL ADAPTIVE RETRIEVER TESTS PASSED")
    print("=" * 80)

    print("\nValidated:")
    print("  ✓ Supported query answered at candidate depth 20")
    print("  ✓ Evidence depth adaptation works")
    print("  ✓ Insufficient evidence triggers retrieval expansion")
    print("  ✓ Candidate depth expands from 20 to 50")
    print("  ✓ Evidence depth follows 1 → 3 → 5 → 10")
    print("  ✓ Out-of-corpus query correctly abstains")
    print("  ✓ AdaptiveDecisionEngine is actively controlling the loop")


if __name__ == "__main__":
    main()
