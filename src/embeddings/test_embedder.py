from embedder import Embedder


embedder = Embedder()

texts = [
    "RAG combines retrieval and generation.",
    "Retrieval augmented generation uses external knowledge.",
    "The weather is sunny today."
]

embeddings = embedder.embed(texts)

print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])