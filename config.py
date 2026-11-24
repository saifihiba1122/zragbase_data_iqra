import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

# Data files
PDF_FILE = DATA_DIR / "modern+database+10th+edition.pdf"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"
INDEX_FILE = DATA_DIR / "faiss.index"
METADATA_FILE = DATA_DIR / "faiss_metadata.json"

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "models/gemini-3.6-flash"

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200

# Search
TOP_K = 10
RELEVANCE_THRESHOLD = 1.4
