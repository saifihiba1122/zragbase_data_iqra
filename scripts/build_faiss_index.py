"""Step 3: Build FAISS index from embeddings and save metadata."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import faiss
import config


def build():
    print("Loading embeddings...")
    with open(config.EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total records: {len(data)}")

    vectors = np.array([item["embedding"] for item in data], dtype=np.float32)
    dimension = vectors.shape[1]
    print(f"Vector shape: {vectors.shape}")

    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    print(f"Vectors indexed: {index.ntotal}")

    faiss.write_index(index, str(config.INDEX_FILE))
    print(f"Index saved -> {config.INDEX_FILE}")

    metadata_records = [
        {
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "metadata": item.get("metadata", {}),
        }
        for item in data
    ]

    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved -> {config.METADATA_FILE}")

    print(f"\nDone! {index.ntotal} vectors, {dimension} dimensions")


if __name__ == "__main__":
    build()
