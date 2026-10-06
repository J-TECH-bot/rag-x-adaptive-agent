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
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.json"
)

PDF_PATH = (
    PROJECT_ROOT / "data" / "raw" / "rag_original.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "evaluation"
    / "evidence_size_diagnostic_512.json"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CANDIDATE_DEPTHS = [20, 50]
EVIDENCE_SIZES = [1, 3, 5, 10]

MODEL_MAX_TOKENS = 512


def build_prompt(question: str, evidence: str) -> str:
    return f"""
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


def combine_evidence(results):
    return "\n\n".join(
        result["text"]
        for result in results
    )


def truncate_for_model(tokenizer, prompt):
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=MODEL_MAX_TOKENS,
        return_tensors="pt"
    )

    return encoded


def main():

    print("=" * 70)
    print("RAG-X CONTROLLED EVIDENCE SIZE DIAGNOSTIC")
    print("Explicit FLAN-T5 input limit: 512 tokens")
    print("=" * 70)

    # ------------------------------------------------
    # Load evaluation questions
    # ------------------------------------------------

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\nQuestions: {len(questions)}")

    # ------------------------------------------------
    # Load and chunk document
    # ------------------------------------------------

    print("\nLoading PDF...")

    text = load_pdf(str(PDF_PATH))

    chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    print(f"Chunks: {len(chunks)}")

    # ------------------------------------------------
    # Embeddings
    # ------------------------------------------------

    print("\nLoading embedder...")

    embedder = Embedder()

    embeddings = embedder.embed(chunks)

    # ------------------------------------------------
    # Vector store
    # ------------------------------------------------

    vector_store = VectorStore(
        embeddings.shape[1]
    )

    vector_store.add(embeddings)

    dense_retriever = Retriever(
        embedder,
        vector_store,
        chunks
    )

    # ------------------------------------------------
    # BM25
    # ------------------------------------------------

    print("\nLoading BM25 retriever...")

    bm25_retriever = BM25Retriever(chunks)

    # ------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever
    )

    # ------------------------------------------------
    # Reranker
    # ------------------------------------------------

    print("\nLoading reranker...")

    reranker = Reranker()

    # ------------------------------------------------
    # Evidence judge
    # ------------------------------------------------

    print("\nLoading evidence judge...")

    evidence_judge = EvidenceSupportJudgeV2()

    tokenizer = evidence_judge.tokenizer

    print("\nVerified model limit:")
    print("Tokenizer max length:", tokenizer.model_max_length)

    # ------------------------------------------------
    # Experiment
    # ------------------------------------------------

    results = []

    total_cases = (
        len(questions)
        * len(CANDIDATE_DEPTHS)
        * len(EVIDENCE_SIZES)
    )

    print(f"\nTotal experiment cases: {total_cases}")

    case_number = 0

    for question_data in questions:

        question_id = question_data["id"]
        question = question_data["question"]

        answerability = question_data["answerability"]
        difficulty_type = question_data["difficulty_type"]

        gold_chunks = set(
            question_data.get("relevant_chunks", [])
        )

        for candidate_depth in CANDIDATE_DEPTHS:

            candidates = hybrid_retriever.retrieve(
                question,
                top_k=candidate_depth,
                candidate_k=candidate_depth
            )

            reranked = reranker.rerank(
                question,
                candidates
            )

            for evidence_size in EVIDENCE_SIZES:

                case_number += 1

                evidence = reranked[:evidence_size]

                evidence_text = combine_evidence(
                    evidence
                )

                prompt = build_prompt(
                    question,
                    evidence_text
                )

                # ----------------------------------------
                # Measure raw prompt length
                # ----------------------------------------

                raw_tokens = tokenizer(
                    prompt,
                    truncation=False,
                    return_tensors=None
                )["input_ids"]

                input_token_count = len(
                    raw_tokens
                )

                # ----------------------------------------
                # Explicit 512-token truncation
                # ----------------------------------------

                encoded = truncate_for_model(
                    tokenizer,
                    prompt
                )

                tokens_after_truncation = (
                    encoded["input_ids"].shape[1]
                )

                was_truncated = (
                    input_token_count
                    > MODEL_MAX_TOKENS
                )

                # ----------------------------------------
                # Run judge using exactly the same
                # 512-token input
                # ----------------------------------------

                outputs = evidence_judge.model.generate(
                    **encoded,
                    max_new_tokens=5,
                    do_sample=False
                )

                raw_output = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                ).strip()

                normalized = raw_output.upper()

                if normalized.startswith("SUPPORTED"):
                    judge_label = "SUPPORTED"
                else:
                    judge_label = "INSUFFICIENT"

                # ----------------------------------------
                # Gold evidence diagnostics
                # ----------------------------------------

                retrieved_chunk_ids = [
                    result["chunk_index"]
                    for result in evidence
                ]

                first_gold_rank = None

                for rank, chunk_id in enumerate(
                    retrieved_chunk_ids,
                    start=1
                ):
                    if chunk_id in gold_chunks:
                        first_gold_rank = rank
                        break

                # ----------------------------------------
                # Save case
                # ----------------------------------------

                record = {
                    "question_id": question_id,
                    "question": question,
                    "answerability": answerability,
                    "difficulty_type": difficulty_type,
                    "candidate_depth": candidate_depth,
                    "evidence_size": evidence_size,

                    "retrieved_chunk_ids":
                        retrieved_chunk_ids,

                    "gold_chunk_ids":
                        sorted(gold_chunks),

                    "first_gold_rank":
                        first_gold_rank,

                    "judge_label":
                        judge_label,

                    "raw_judge_output":
                        raw_output,

                    "input_token_count":
                        input_token_count,

                    "tokens_after_truncation":
                        tokens_after_truncation,

                    "was_truncated":
                        was_truncated,

                    "model_max_tokens":
                        MODEL_MAX_TOKENS
                }

                results.append(record)

                if case_number % 40 == 0:
                    print(
                        f"Progress: "
                        f"{case_number}/{total_cases}"
                    )

    # ------------------------------------------------
    # Save results
    # ------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("CONTROLLED DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(f"Total records: {len(results)}")
    print(f"Output: {OUTPUT_PATH}")

    # ------------------------------------------------
    # Basic truncation summary
    # ------------------------------------------------

    truncated = sum(
        1
        for x in results
        if x["was_truncated"]
    )

    print(
        f"Truncated records: "
        f"{truncated}/{len(results)}"
    )

    print(
        f"Truncation rate: "
        f"{100 * truncated / len(results):.1f}%"
    )


if __name__ == "__main__":
    main()
