from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embedder import Embedder


# 1. Load PDF
pdf_path = "data/raw/rag_original.pdf"

text = load_pdf(pdf_path)

print("Total characters:", len(text))


# 2. Create chunks
chunks = chunk_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
)

print("Total chunks:", len(chunks))


# 3. Create embedding model
embedder = Embedder()


# 4. Convert all chunks into vectors
embeddings = embedder.embed(chunks)

print("Embedding shape:", embeddings.shape)