"""Step 1: Extract text from PDF and split into chunks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from pypdf import PdfReader
import config


def create_chunks():
    if not config.PDF_FILE.exists():
        print(f"PDF not found: {config.PDF_FILE}")
        return

    pdf = PdfReader(str(config.PDF_FILE))
    full_text = ""
    for page in pdf.pages:
        full_text += (page.extract_text() or "") + "\n"
    full_text = full_text.strip()

    if not full_text:
        print("No text found in PDF.")
        return

    chunks = []
    start = 0
    chunk_id = 1

    while start < len(full_text):
        end = min(start + config.CHUNK_SIZE, len(full_text))
        text = full_text[start:end].strip()

        if text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "source": config.PDF_FILE.name,
                    "char_count": len(text),
                },
            })
            chunk_id += 1

        if end >= len(full_text):
            break
        start = end - config.CHUNK_OVERLAP

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(chunks)} chunks -> {config.CHUNKS_FILE}")


if __name__ == "__main__":
    create_chunks()
