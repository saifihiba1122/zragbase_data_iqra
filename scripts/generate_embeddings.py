"""Step 2: Generate embeddings for each chunk using SentenceTransformer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from sentence_transformers import SentenceTransformer
import config


def generate():
    print("Loading chunks...")
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True)

    results = []
    for chunk, vector in zip(chunks, vectors):
        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "embedding": vector.tolist(),
        })

    with open(config.EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"Saved {len(results)} embeddings ({len(vectors[0])}d) -> {config.EMBEDDINGS_FILE}")


if __name__ == "__main__":
    generate()
