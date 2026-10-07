import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text

from embeddings.embedder import Embedder

from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

from evaluation.evidence_support_v3_1 import EvidenceSupportJudgeV3_1


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"

TARGET_IDS = {54, 56, 57, 58, 59, 60}
EVIDENCE_DEPTHS = [1, 3, 5, 10]


def build_retrieval_pipeline(chunks):
    print("\nBuilding retrieval pipeline...")

    embedder = Embedder()

    embeddings = embedder.embed(chunks)

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(embeddings)

    dense_retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks
    )

    bm25_retriever = BM25Retriever(chunks)

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever
    )

    reranker = Reranker()

    judge = EvidenceSupportJudgeV3_1()

    return hybrid_retriever, reranker, judge


def main():

    print("=" * 70)
    print("RAG-X V3 FALSE-SUPPORT DIAGNOSTIC")
    print("=" * 70)

    print("\nLoading PDF...")

    text = load_pdf(PDF_PATH)

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Chunks: {len(chunks)}")

    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    target_questions = [
        q for q in questions
        if q["id"] in TARGET_IDS
    ]

    target_questions.sort(
        key=lambda x: x["id"]
    )

    print(
        f"Testing questions: "
        f"{[q['id'] for q in target_questions]}"
    )

    hybrid_retriever, reranker, judge = build_retrieval_pipeline(
        chunks
    )

    for question_data in target_questions:

        question_id = question_data["id"]
        question = question_data["question"]

        print("\n")
        print("=" * 70)
        print(f"Q{question_id}")
        print("=" * 70)

        print(f"\nQuestion:")
        print(question)

        print(
            f"\nAnswerability: "
            f"{question_data['answerability']}"
        )

        print(
            f"Difficulty: "
            f"{question_data['difficulty_type']}"
        )

        print(
            f"Relevant chunks: "
            f"{question_data['relevant_chunks']}"
        )

        print("\nRetrieving candidate pool...")

        candidates = hybrid_retriever.retrieve(
            question,
            top_k=20,
            candidate_k=20
        )

        print(
            f"Hybrid candidates: "
            f"{len(candidates)}"
        )

        print("\nReranking candidates...")

        reranked = reranker.rerank(
            question,
            candidates
        )

        print("\nTop reranked candidates:")

        for rank, result in enumerate(
            reranked[:10],
            start=1
        ):
            print(
                f"\nRank {rank}"
                f" | Chunk {result['chunk_index']}"
                f" | Reranker score "
                f"{result['reranker_score']:.4f}"
            )

            print(
                result["text"][:500]
                .replace("\n", " ")
            )

        print("\n")
        print("-" * 70)
        print("V3 EVIDENCE JUDGMENTS")
        print("-" * 70)

        for depth in EVIDENCE_DEPTHS:

            selected = reranked[:depth]

            evidence = "\n\n".join(
                result["text"]
                for result in selected
            )

            result = judge.judge(
                question,
                evidence
            )

            print(
                f"\nEvidence depth: Top-{depth}"
            )

            print(
                f"V3 label: {result['label']}"
            )

            print(
                f"Raw output: {result['raw_output']}"
            )


if __name__ == "__main__":
    main()
