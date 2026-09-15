import json

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever


PDF_PATH = "data/raw/rag_original.pdf"
QUESTIONS_PATH = "data/evaluation/retrieval_questions.json"


def has_relevant_chunk(retrieved_indices, relevant_chunks):
    return bool(
        set(retrieved_indices) & set(relevant_chunks)
    )


def main():

    # --------------------------------------------------
    # 1. Load PDF and create the same chunks
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
    # 4. Load evaluation questions
    # --------------------------------------------------

    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    dense_hits = []
    bm25_hits = []
    union_hits = []

    print("\n" + "=" * 75)
    print("DENSE vs BM25 vs UNION — CANDIDATE COVERAGE")
    print("=" * 75)

    # --------------------------------------------------
    # 5. Evaluate candidate coverage
    # --------------------------------------------------

    for item in questions:

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        dense_results = dense_retriever.retrieve(
            question,
            top_k=20
        )

        bm25_results = bm25_retriever.retrieve(
            question,
            top_k=20
        )

        dense_indices = [
            result["chunk_index"]
            for result in dense_results
        ]

        bm25_indices = [
            result["chunk_index"]
            for result in bm25_results
        ]

        union_indices = sorted(
            set(dense_indices) | set(bm25_indices)
        )

        dense_found = has_relevant_chunk(
            dense_indices,
            relevant_chunks
        )

        bm25_found = has_relevant_chunk(
            bm25_indices,
            relevant_chunks
        )

        union_found = has_relevant_chunk(
            union_indices,
            relevant_chunks
        )

        dense_hits.append(int(dense_found))
        bm25_hits.append(int(bm25_found))
        union_hits.append(int(union_found))

        print(f"\nQuestion: {question}")
        print(f"Relevant chunks: {relevant_chunks}")

        print(
            f"Dense found: {dense_found}"
        )

        print(
            f"BM25 found:  {bm25_found}"
        )

        print(
            f"Union found: {union_found}"
        )

        print(
            f"Dense Top-20: {dense_indices}"
        )

        print(
            f"BM25 Top-20:  {bm25_indices}"
        )

        print(
            f"Union size:   {len(union_indices)}"
        )

    # --------------------------------------------------
    # 6. Summary
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("CANDIDATE COVERAGE SUMMARY")
    print("=" * 75)

    print(
        f"Dense Top-20 coverage: "
        f"{sum(dense_hits) / len(dense_hits):.3f}"
    )

    print(
        f"BM25 Top-20 coverage:  "
        f"{sum(bm25_hits) / len(bm25_hits):.3f}"
    )

    print(
        f"Union coverage:        "
        f"{sum(union_hits) / len(union_hits):.3f}"
    )


if __name__ == "__main__":
    main()