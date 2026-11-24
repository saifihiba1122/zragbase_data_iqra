import json
import re
import time

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

import config

# Load resources once at import
print("Loading FAISS index...")
index = faiss.read_index(str(config.INDEX_FILE))
print(f"FAISS loaded: {index.ntotal} vectors")

print("Loading metadata...")
with open(config.METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"Metadata loaded: {len(metadata)} records")

print("Loading embedding model...")
embedder = SentenceTransformer(config.EMBEDDING_MODEL)
print("Embedding model ready!")

gemini_client = None
if config.GEMINI_API_KEY and config.GEMINI_API_KEY != "YOUR_API_KEY_HERE":
    gemini_client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=30000),
    )
    print("Gemini client connected!")
else:
    print("WARNING: GEMINI_API_KEY not set. LLM generation disabled.")

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in",
    "on", "to", "for", "and", "explain", "define",
    "describe", "how", "which", "why", "does", "do",
    "can", "this", "that", "it", "with", "from", "by",
}


def embed_query(query: str) -> np.ndarray:
    vector = embedder.encode(query, convert_to_numpy=True)
    return np.asarray([vector], dtype=np.float32)


def search_faiss(query_vector: np.ndarray, top_k: int = config.TOP_K):
    candidate_k = min(max(top_k * 5, 50), index.ntotal)
    distances, indices = index.search(query_vector, candidate_k)
    return distances[0], indices[0]


def rank_chunks(query: str, distances, indices, top_k: int = config.TOP_K):
    query_words = set(re.findall(r"\b\w+\b", query.lower())) - STOP_WORDS

    results = []
    for rank, idx in enumerate(indices):
        if idx < 0 or idx >= len(metadata):
            continue

        record = metadata[idx]
        text = record.get("text", "")
        doc_words = set(re.findall(r"\b\w+\b", text.lower()))

        keyword_score = 0.0
        if query_words:
            keyword_score = len(query_words & doc_words) / len(query_words)

        semantic_score = 1 / (rank + 1)
        final_score = 0.65 * semantic_score + 0.35 * keyword_score

        results.append({
            "chunk_id": record.get("chunk_id", idx),
            "distance": float(distances[rank]),
            "keyword_score": keyword_score,
            "final_score": final_score,
            "text": text,
            "metadata": record.get("metadata", {}),
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for i, r in enumerate(results[:top_k]):
        r["rank"] = i + 1
    return results[:top_k]


def is_relevant(results: list) -> bool:
    if not results:
        return False
    return results[0]["distance"] < config.RELEVANCE_THRESHOLD


def get_source_label(source: dict) -> str:
    info = source.get("metadata", {})
    page = info.get("page") or info.get("page_number") or info.get("page_num")
    label = f"Chunk {source['chunk_id']}"
    if page is not None:
        label += f" | Page {page}"
    return label


def generate_with_llm(query: str, chunks: list) -> tuple[str, str | None]:
    if gemini_client is None:
        return "", "GEMINI_API_KEY is not configured. Set it in .env file."

    context = ""
    for chunk in chunks:
        label = get_source_label(chunk)
        context += f"\n\n[{label}]\n{chunk['text']}"

    prompt = f"""You are a helpful book assistant for "Modern Database Management, 10th Edition".

Answer the question using ONLY the provided book context below.

Rules:
1. Give a clear, well-structured answer.
2. Do NOT invent any information beyond what the context provides.
3. If the context is insufficient, say so honestly.
4. Reference chunk IDs when relevant.

BOOK CONTEXT:
{context}

QUESTION:
{query}"""

    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
            )
            if not response.text:
                return "", "Gemini returned an empty response."
            return response.text, None
        except Exception as e:
            error_str = str(e)
            if ("503" in error_str or "UNAVAILABLE" in error_str or "429" in error_str) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return "", error_str


def pipeline(query: str) -> dict:
    start = time.time()

    query_vector = embed_query(query)
    embed_time = time.time() - start

    t = time.time()
    distances, indices = search_faiss(query_vector)
    search_time = time.time() - t

    t = time.time()
    chunks = rank_chunks(query, distances, indices)
    rank_time = time.time() - t

    relevant = is_relevant(chunks)

    answer = ""
    llm_error = None
    llm_time = 0.0

    if relevant:
        t = time.time()
        answer, llm_error = generate_with_llm(query, chunks[:5])
        llm_time = time.time() - t

    total_time = time.time() - start

    return {
        "query": query,
        "answer": answer,
        "chunks": chunks,
        "relevant": relevant,
        "llm_error": llm_error,
        "timing": {
            "embed": embed_time,
            "search": search_time,
            "rank": rank_time,
            "llm": llm_time,
            "total": total_time,
        },
    }


def search_database(query: str, top_k: int = config.TOP_K) -> list:
    query_vector = embed_query(query)
    distances, indices = search_faiss(query_vector, top_k)
    return rank_chunks(query, distances, indices, top_k)


if __name__ == "__main__":
    print("\nDATABASE BOOK RAG SYSTEM")
    print("Type 'exit' to close.\n")

    while True:
        query = input("Your Question: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        result = pipeline(query)

        if not result["relevant"]:
            print("\nThis question is outside the book's knowledge base.\n")
            continue

        if result["llm_error"]:
            print(f"\nLLM Error: {result['llm_error']}\n")
        else:
            print(f"\n{result['answer']}\n")

        t = result["timing"]
        print(f"[Embed: {t['embed']:.2f}s | Search: {t['search']:.2f}s | LLM: {t['llm']:.2f}s | Total: {t['total']:.2f}s]")
