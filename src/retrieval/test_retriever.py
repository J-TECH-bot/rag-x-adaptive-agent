from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever


print("STEP 1: Loading PDF", flush=True)

pdf_path = "data/raw/rag_original.pdf"
text = load_pdf(pdf_path)

print("STEP 1 DONE:", len(text), "characters", flush=True)


print("STEP 2: Chunking", flush=True)

chunks = chunk_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
)

print("STEP 2 DONE:", len(chunks), "chunks", flush=True)


print("STEP 3: Loading embedding model", flush=True)

embedder = Embedder()

print("STEP 3 DONE", flush=True)


print("STEP 4: Creating document embeddings", flush=True)

document_embeddings = embedder.embed(chunks)

print("STEP 4 DONE:", document_embeddings.shape, flush=True)


print("STEP 5: Creating vector store", flush=True)

dimension = document_embeddings.shape[1]

vector_store = VectorStore(dimension)

vector_store.add(document_embeddings)

print("STEP 5 DONE:", vector_store.index.ntotal, "vectors", flush=True)


print("STEP 6: Creating Retriever", flush=True)

retriever = Retriever(
    embedder=embedder,
    vector_store=vector_store,
    chunks=chunks
)

print("STEP 6 DONE", flush=True)


print("STEP 7: Querying", flush=True)

query = "What is Retrieval-Augmented Generation?"

results = retriever.retrieve(
    query,
    top_k=5
)

print("STEP 7 DONE", flush=True)


print("\nRetrieved results:\n", flush=True)

for rank, result in enumerate(results, start=1):

    print(f"Rank: {rank}", flush=True)
    print(f"Chunk index: {result['chunk_index']}", flush=True)
    print(f"Score: {result['score']:.4f}", flush=True)
    print(f"Text: {result['text'][:500]}", flush=True)
    print("-" * 80, flush=True)