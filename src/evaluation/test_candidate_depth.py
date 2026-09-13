import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"


def recall_at_k(results, relevant_chunks, k):
    retrieved_chunks = {
        result["chunk_index"]
        for result in results[:k]
    }

    relevant_chunks = set(relevant_chunks)

    return int(bool(retrieved_chunks & relevant_chunks))


def main():

    # -----------------------------
    # 1. Load PDF
    # -----------------------------
    text = load_pdf(PDF_PATH)

    # -----------------------------
    # 2. Create chunks
    # -----------------------------
    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Characters: {len(text)}")
    print(f"Chunks: {len(chunks)}")

    # -----------------------------
    # 3. Create embeddings
    # -----------------------------
    embedder = Embedder()

    embeddings = embedder.embed(chunks)

    print(f"Embeddings: {embeddings.shape}")

    # -----------------------------
    # 4. Create vector store
    # -----------------------------
    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(embeddings)

    print(f"Vectors stored: {len(chunks)}")

    # -----------------------------
    # 5. Create retriever
    # -----------------------------
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        chunks=chunks
    )

    # -----------------------------
    # 6. Load evaluation questions
    # -----------------------------
    with open(QUESTIONS_PATH, "r") as file:
        questions = json.load(file)

    k_values = [5, 10, 20, 50]

    totals = {
        k: 0
        for k in k_values
    }

    # -----------------------------
    # 7. Evaluate
    # -----------------------------
    for item in questions:

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        results = retriever.retrieve(
            question,
            top_k=50
        )

        print("\nQuestion:")
        print(question)

        print(
            "Relevant chunks:",
            relevant_chunks
        )

        print(
            "Retrieved Top-10:",
            [
                result["chunk_index"]
                for result in results[:10]
            ]
        )

        for k in k_values:

            score = recall_at_k(
                results,
                relevant_chunks,
                k
            )

            totals[k] += score

            print(
                f"Recall@{k}: {score}"
            )

    # -----------------------------
    # 8. Final results
    # -----------------------------
    print("\n")
    print("=" * 60)
    print("CANDIDATE DEPTH EVALUATION")
    print("=" * 60)

    total_questions = len(questions)

    for k in k_values:

        recall = totals[k] / total_questions

        print(
            f"Recall@{k}: {recall:.3f}"
        )


if __name__ == "__main__":
    main()