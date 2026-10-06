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

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_questions.json"
)

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rag_original.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evidence_size_diagnostic.json"
)


def combine_evidence(results):
    return "\n\n".join(
        result["text"]
        for result in results
    )


def get_first_gold_rank(
    reranked_results,
    gold_chunks
):
    gold_chunks = set(gold_chunks)

    for rank, result in enumerate(
        reranked_results,
        start=1
    ):
        if result["chunk_index"] in gold_chunks:
            return rank

    return None


def analyze_tokenization(
    tokenizer,
    question,
    evidence
):
    prompt = f"""
You are an evidence verification system.

Your task is to determine whether the EVIDENCE
contains enough information to answer the QUESTION.

Return exactly ONE label:

SUPPORTED
INSUFFICIENT

SUPPORTED means:
The evidence explicitly contains the information
needed to answer the question.

INSUFFICIENT means:
The evidence does not contain enough information
to answer the question.

Important rules:

1. Related information is NOT enough.
2. Do not use your own outside knowledge.
3. Do not assume information that is not written
   in the evidence.
4. If the answer cannot be determined directly
   from the evidence, return INSUFFICIENT.
5. A question about a fact that is absent from
   the evidence must be INSUFFICIENT.

QUESTION:
{question}

EVIDENCE:
{evidence}

LABEL:
"""

    untruncated = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False
    )

    truncated = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    untruncated_tokens = len(
        untruncated["input_ids"][0]
    )

    truncated_tokens = len(
        truncated["input_ids"][0]
    )

    return {
        "input_token_count": untruncated_tokens,
        "tokens_after_truncation": truncated_tokens,
        "was_truncated": (
            untruncated_tokens > 2048
        )
    }


def main():

    print("=" * 60)
    print("RAG-X Evidence Size Diagnostic")
    print("=" * 60)

    # --------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------

    with open(
        QUESTIONS_PATH,
        "r"
    ) as f:
        questions = json.load(f)

    print(
        f"Evaluation questions: {len(questions)}"
    )

    # --------------------------------------------------
    # Load PDF
    # --------------------------------------------------

    print("\nLoading PDF...")

    text = load_pdf(str(PDF_PATH))

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    print("\nLoading embedding model...")

    embedder = Embedder()

    embeddings = embedder.embed(
        chunks
    )

    # --------------------------------------------------
    # Vector store
    # --------------------------------------------------

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(
        embeddings
    )

    dense_retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks
    )

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    print("\nBuilding BM25 retriever...")

    bm25_retriever = BM25Retriever(
        chunks
    )

    # --------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever
    )

    # --------------------------------------------------
    # Reranker
    # --------------------------------------------------

    print("\nLoading reranker...")

    reranker = Reranker()

    # --------------------------------------------------
    # Evidence judge
    # --------------------------------------------------

    print("\nLoading evidence judge...")

    evidence_judge = EvidenceSupportJudgeV2()

    tokenizer = evidence_judge.tokenizer

    # --------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------

    candidate_depths = [20, 50]

    evidence_sizes = [
        1,
        3,
        5,
        10
    ]

    results = []

    total_experiments = (
        len(questions)
        * len(candidate_depths)
        * len(evidence_sizes)
    )

    experiment_number = 0

    print(
        f"\nTotal experiment cases: "
        f"{total_experiments}"
    )

    # --------------------------------------------------
    # Main experiment
    # --------------------------------------------------

    for question_data in questions:

        question_id = question_data["id"]
        question = question_data["question"]

        answerability = (
            question_data["answerability"]
        )

        difficulty_type = (
            question_data["difficulty_type"]
        )

        gold_chunks = (
            question_data["relevant_chunks"]
        )

        print(
            f"\nQuestion {question_id}/"
            f"{len(questions)}: {question}"
        )

        for candidate_depth in candidate_depths:

            candidates = (
                hybrid_retriever.retrieve(
                    question,
                    top_k=candidate_depth,
                    candidate_k=candidate_depth
                )
            )

            reranked = reranker.rerank(
                question,
                candidates
            )

            first_gold_rank = (
                get_first_gold_rank(
                    reranked,
                    gold_chunks
                )
            )

            for evidence_size in evidence_sizes:

                experiment_number += 1

                evidence = reranked[
                    :evidence_size
                ]

                evidence_chunk_ids = [
                    result["chunk_index"]
                    for result in evidence
                ]

                evidence_text = combine_evidence(
                    evidence
                )

                token_info = analyze_tokenization(
                    tokenizer,
                    question,
                    evidence_text
                )

                judgment = evidence_judge.judge(
                    question,
                    evidence_text
                )

                result = {
                    "question_id": question_id,
                    "question": question,
                    "answerability": answerability,
                    "difficulty_type": difficulty_type,

                    "candidate_depth": candidate_depth,

                    "evidence_size": evidence_size,

                    "retrieved_chunk_ids": (
                        evidence_chunk_ids
                    ),

                    "gold_chunk_ids": gold_chunks,

                    "first_gold_rank": (
                        first_gold_rank
                    ),

                    "judge_label": (
                        judgment["label"]
                    ),

                    "raw_judge_output": (
                        judgment["raw_output"]
                    ),

                    "input_token_count": (
                        token_info[
                            "input_token_count"
                        ]
                    ),

                    "tokens_after_truncation": (
                        token_info[
                            "tokens_after_truncation"
                        ]
                    ),

                    "was_truncated": (
                        token_info[
                            "was_truncated"
                        ]
                    )
                }

                results.append(result)

                print(
                    f"  depth={candidate_depth:2d} "
                    f"evidence={evidence_size:2d} "
                    f"judge={judgment['label']:12s} "
                    f"tokens={token_info['input_token_count']:4d} "
                    f"truncated="
                    f"{token_info['was_truncated']}"
                )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print("\n" + "=" * 60)
    print("Diagnostic complete.")
    print("=" * 60)

    print(
        f"Results saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nTotal records: {len(results)}"
    )


if __name__ == "__main__":
    main()
