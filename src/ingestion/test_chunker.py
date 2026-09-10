from loader import load_pdf
from chunker import chunk_text


pdf_path = "data/raw/rag_original.pdf"

# Step 1: Extract text from PDF
text = load_pdf(pdf_path)

print("Total characters:", len(text))


# Step 2: Split text into chunks
chunks = chunk_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
)

print("Total chunks:", len(chunks))


# Step 3: Inspect first 3 chunks
for i, chunk in enumerate(chunks[:3]):

    print(f"\n--- Chunk {i} ---")
    print("Characters:", len(chunk))
    print(chunk[:500])