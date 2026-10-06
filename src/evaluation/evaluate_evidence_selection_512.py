import json
from pathlib import Path

from transformers import AutoTokenizer

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from evaluation.evidence_budget import EvidenceBudget
from evaluation.evidence_support_v2 import EvidenceSupportJudgeV2


# ============================================================
# Configuration
# ============================================================

QUESTIONS_PATH = Path(
    "data/evaluation/retrieval_questions.json"
)

PDF_PATH = Path(
    "data/raw/rag_original.pdf"
)

OUTPUT_PATH = Path(
    "data/evaluation/evidence_selection_512.json"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
JUDGE_MODEL = "google/flan-t5-base"

CANDIDATE_DEPTH = 50
MODEL_MAX_TOKENS = 512

SELECTION_METHODS = [
    "top_1",
    "top_3",
    "top_5",
    "top_10",
    "token_budget"
]


# ============================================================
# Helpers
# ============================================================

def combine_evidence(results):
    return "\n\n".join(
        result["text"]
        for result in results
    )


def build_prompt(question, evidence_text):
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
{evidence_text}

LABEL:
"""


def truncate_for_model(tokenizer, prompt):
    return tokenizer(
        prompt,
        truncation=True,
        max_length=MODEL_MAX_TOKENS,
        return_tensors="pt"
    )


def select_evidence(
    method,
    reranked,
    evidence_budget
):
    if method == "top_1":
        return reranked[:1]

    if method == "top_3":
        return reranked[:3]

    if method == "top_5":
        return reranked[:5]

    if method == "top_10":
        return reranked[:10]

    if method == "token_budget":
        result = evidence_budget.select(reranked)

        return result.selected_results

    raise ValueError(
        f"Unknown selection method: {method}"
    )


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("RAG-X EVIDENCE SELECTION EXPERIMENT")
print("=" * 70)

questions = json.loads(
    QUESTIONS_PATH.read_text()
)

print(f"Questions: {len(questions)}")


# ============================================================
# Load document
# ============================================================

print("\nLoading PDF...")

text = load_pdf(str(PDF_PATH))

chunks = chunk_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Chunks: {len(chunks)}")


# ============================================================
# Dense retrieval
# ============================================================

print("\nLoading embedding model...")

embedder = Embedder(
    model_name=EMBEDDING_MODEL
)

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


# ============================================================
# BM25
# ============================================================

print("\nLoading BM25...")

bm25_retriever = BM25Retriever(
    chunks=chunks
)


# ============================================================
# Hybrid retrieval
# ============================================================

hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    bm25_retriever=bm25_retriever
)


# ============================================================
# Reranker
# ============================================================

print("\nLoading reranker...")

reranker = Reranker(
    model_name=RERANKER_MODEL
)


# ============================================================
# Evidence judge
# ============================================================

print("\nLoading evidence judge...")

evidence_judge = EvidenceSupportJudgeV2(
    model_name=JUDGE_MODEL
)

tokenizer = evidence_judge.tokenizer


# ============================================================
# Evidence budget
# ============================================================

evidence_budget = EvidenceBudget(
    tokenizer=tokenizer,
    max_tokens=MODEL_MAX_TOKENS
)


# ============================================================
# Experiment
# ============================================================

results = []

total_cases = (
    len(questions)
    * len(SELECTION_METHODS)
)

case_number = 0

print("\nStarting experiment...")
print(f"Total cases: {total_cases}")


for question_data in questions:

    question_id = question_data["id"]
    question = question_data["question"]

    answerability = question_data["answerability"]
    difficulty_type = question_data["difficulty_type"]

    gold_chunks = set(
        question_data.get("relevant_chunks", [])
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Retrieval and reranking happen ONCE per question.
    # Every selection method receives exactly the same
    # reranked candidate list.
    # --------------------------------------------------------

    candidates = hybrid_retriever.retrieve(
        question,
        top_k=CANDIDATE_DEPTH,
        candidate_k=CANDIDATE_DEPTH
    )

    reranked = reranker.rerank(
        question,
        candidates
    )

    for method in SELECTION_METHODS:

        case_number += 1

        evidence = select_evidence(
            method,
            reranked,
            evidence_budget
        )

        evidence_text = combine_evidence(
            evidence
        )

        prompt = build_prompt(
            question,
            evidence_text
        )

        # ----------------------------------------------------
        # Raw prompt token count
        # ----------------------------------------------------

        raw_tokens = tokenizer(
            prompt,
            truncation=False,
            return_tensors=None
        )["input_ids"]

        input_token_count = len(
            raw_tokens
        )

        # ----------------------------------------------------
        # Explicit 512-token model input
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Judge
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Evidence diagnostics
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Actual evidence token count
        # ----------------------------------------------------

        evidence_token_count = len(
            tokenizer(
                evidence_text,
                truncation=False,
                add_special_tokens=False,
                return_tensors=None
            )["input_ids"]
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        record = {
            "question_id": question_id,
            "question": question,

            "answerability": answerability,
            "difficulty_type": difficulty_type,

            "candidate_depth": CANDIDATE_DEPTH,

            "selection_method": method,

            "evidence_count": len(evidence),

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

            "evidence_token_count":
                evidence_token_count,

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

        if case_number % 20 == 0:
            print(
                f"Progress: "
                f"{case_number}/{total_cases}"
            )


# ============================================================
# Save
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH.write_text(
    json.dumps(
        results,
        indent=2
    )
)

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(f"Total records: {len(results)}")
print(f"Saved to: {OUTPUT_PATH}")

for method in SELECTION_METHODS:

    method_results = [
        r for r in results
        if r["selection_method"] == method
    ]

    supported = sum(
        r["judge_label"] == "SUPPORTED"
        for r in method_results
    )

    truncated = sum(
        r["was_truncated"]
        for r in method_results
    )

    print(
        f"{method:15s} "
        f"SUPPORTED={supported:2d}/{len(method_results)} "
        f"TRUNCATED={truncated:3d}"
    )
