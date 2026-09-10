from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore


# 1. Load document
pdf_path = "data/raw/rag_original.pdf"

text = load_pdf(pdf_path)

print("Characters:", len(text))


# 2. Create chunks
chunks = chunk_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
)

print("Chunks:", len(chunks))


# 3. Create embeddings
embedder = Embedder()

document_embeddings = embedder.embed(chunks)

print("Embeddings:", document_embeddings.shape)


# 4. Create FAISS index
dimension = document_embeddings.shape[1]

vector_store = VectorStore(dimension)

vector_store.add(document_embeddings)

print("Vectors stored:", vector_store.index.ntotal)


# 5. Create query
query = "What is Retrieval-Augmented Generation?"

query_embedding = embedder.embed([query])

print("Query embedding:", query_embedding.shape)


# 6. Search
scores, indices = vector_store.search(
    query_embedding,
    top_k=5
)


# 7. Display results
print("\nTop 5 results:\n")

for rank, (score, index) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    print(f"Rank {rank}")
    print(f"Chunk index: {index}")
    print(f"Similarity score: {score:.4f}")
    print(f"Text: {chunks[index][:500]}")
    print("-" * 80)