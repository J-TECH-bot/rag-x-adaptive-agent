import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever

from evaluation.evaluator import recall_at_k, reciprocal_rank


# 1. Load evaluation questions
with open(
    "data/evaluation/retrieval_questions.json",
    "r"
) as f:
    questions = json.load(f)


# 2. Load document
text = load_pdf("data/raw/rag_original.pdf")

print("Characters:", len(text))


# 3. Create chunks
chunks = chunk_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
)

print("Chunks:", len(chunks))


# 4. Create embeddings
embedder = Embedder()

document_embeddings = embedder.embed(chunks)

print("Embeddings:", document_embeddings.shape)


# 5. Create vector store
dimension = document_embeddings.shape[1]

vector_store = VectorStore(dimension)

vector_store.add(document_embeddings)

print("Vectors stored:", vector_store.index.ntotal)


# 6. Create retriever
retriever = Retriever(
    embedder=embedder,
    vector_store=vector_store,
    chunks=chunks
)


# 7. Evaluate
recall_1_scores = []
recall_3_scores = []
recall_5_scores = []
mrr_scores = []


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
        k=1
    )

    r3 = recall_at_k(
        results,
        relevant_chunks,
        k=3
    )

    r5 = recall_at_k(
        results,
        relevant_chunks,
        k=5
    )

    mrr = reciprocal_rank(
        results,
        relevant_chunks
    )

    recall_1_scores.append(r1)
    recall_3_scores.append(r3)
    recall_5_scores.append(r5)
    mrr_scores.append(mrr)

    print("\nQuestion:")
    print(question)

    print("Relevant chunks:", relevant_chunks)

    print(
        "Retrieved:",
        [r["chunk_index"] for r in results]
    )

    print(f"Recall@1: {r1}")
    print(f"Recall@3: {r3}")
    print(f"Recall@5: {r5}")
    print(f"MRR: {mrr:.3f}")


# 8. Calculate averages

num_questions = len(questions)

avg_recall_1 = sum(recall_1_scores) / num_questions
avg_recall_3 = sum(recall_3_scores) / num_questions
avg_recall_5 = sum(recall_5_scores) / num_questions
avg_mrr = sum(mrr_scores) / num_questions


# 9. Final results

print("\n" + "=" * 60)
print("BASELINE RETRIEVAL EVALUATION")
print("=" * 60)

print(f"Questions: {num_questions}")
print(f"Recall@1: {avg_recall_1:.3f}")
print(f"Recall@3: {avg_recall_3:.3f}")
print(f"Recall@5: {avg_recall_5:.3f}")
print(f"MRR:      {avg_mrr:.3f}")