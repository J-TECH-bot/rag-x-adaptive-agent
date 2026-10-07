import json
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = PROJECT_ROOT / "data/raw/rag_original.pdf"
QUESTIONS_PATH = PROJECT_ROOT / "data/evaluation/retrieval_questions.json"

TARGET_IDS = {54, 56, 57, 58, 59, 60}

CANDIDATE_K = 20
EVIDENCE_DEPTHS = [1, 3, 5, 10]


def build_system():

    print("=" * 70)
    print("BUILDING RAG-X DIAGNOSTIC SYSTEM")
    print("=" * 70)

    print("\nLoading PDF...")
    text = load_pdf(str(PDF_PATH))

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Chunks: {len(chunks)}")

    print("\nLoading embedder...")
    embedder = Embedder()

    print("\nBuilding vector store...")
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

    print("\nBuilding BM25 retriever...")
    bm25_retriever = BM25Retriever(chunks)

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever
    )

    print("\nLoading reranker...")
    reranker = Reranker()

    print("\nLoading evidence judge...")
    evidence_judge = EvidenceSupportJudgeV2()

    print("\nSystem ready.")

    return (
        chunks,
        hybrid_retriever,
        reranker,
        evidence_judge
    )


def print_result_block(
    question_data,
    chunks,
    hybrid_retriever,
    reranker,
    evidence_judge
):

    question_id = question_data["id"]
    question = question_data["question"]

    print("\n")
    print("=" * 100)
    print(f"QUESTION {question_id}")
    print("=" * 100)

    print("\nQuestion:")
    print(question)

    print("\nAnswerability:")
    print(question_data["answerability"])

    print("\nDifficulty:")
    print(question_data["difficulty_type"])

    print("\nReference answer:")
    print(question_data["reference_answer"])

    gold_chunks = question_data.get(
        "relevant_chunks",
        []
    )

    print("\nGold relevant chunks:")
    print(gold_chunks)

    print("\n" + "-" * 100)
    print("RETRIEVAL + RERANKING")
    print("-" * 100)

    candidates = hybrid_retriever.retrieve(
        question,
        top_k=CANDIDATE_K,
        candidate_k=CANDIDATE_K
    )

    reranked = reranker.rerank(
        question,
        candidates
    )

    retrieved_chunk_ids = {
        r["chunk_index"]
        for r in candidates
    }

    reached_gold_chunks = sorted(
        set(gold_chunks) & retrieved_chunk_ids
    )

    print(
        f"\nRetrieved candidates: {len(candidates)}"
    )

    print(
        "Relevant chunk IDs reached: "
        f"{reached_gold_chunks}"
    )

    print("\nReranked top 20:")

    for rank, result in enumerate(
        reranked,
        start=1
    ):

        print(
            f"\nRANK {rank}"
            f" | chunk={result['chunk_index']}"
            f" | retrieval_score={result['score']:.6f}"
            f" | reranker_score={result['reranker_score']:.6f}"
        )

        print(result["text"][:700])

    print("\n" + "-" * 100)
    print("EVIDENCE JUDGE DIAGNOSTIC")
    print("-" * 100)

    for depth in EVIDENCE_DEPTHS:

        selected = reranked[:depth]

        selected_chunk_ids = [
            r["chunk_index"]
            for r in selected
        ]

        evidence_text = "\n\n".join(
            [
                f"[CHUNK {r['chunk_index']}]\n{r['text']}"
                for r in selected
            ]
        )

        print("\n" + "=" * 80)
        print(f"EVIDENCE DEPTH = {depth}")
        print("=" * 80)

        print("\nSelected chunk IDs:")
        print(selected_chunk_ids)

        print("\nJudge input evidence:")
        print("-" * 80)
        print(evidence_text)
        print("-" * 80)

        result = evidence_judge.judge(
            question,
            evidence_text
        )

        print("\nJUDGE OUTPUT:")
        print(f"Label: {result['label']}")
        print(f"Raw output: {result['raw_output']}")

    print("\n" + "=" * 100)
    print(f"END QUESTION {question_id}")
    print("=" * 100)


def main():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        questions = json.load(f)

    target_questions = [
        q
        for q in questions
        if int(q["id"]) in TARGET_IDS
    ]

    print(
        f"Found {len(target_questions)} target questions."
    )

    (
        chunks,
        hybrid_retriever,
        reranker,
        evidence_judge
    ) = build_system()

    for question_data in target_questions:

        print_result_block(
            question_data,
            chunks,
            hybrid_retriever,
            reranker,
            evidence_judge
        )


if __name__ == "__main__":
    main()
