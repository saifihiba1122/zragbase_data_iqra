# RAG-Based Database Book Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions using content extracted from *Modern Database Management, 10th Edition*.

## Project Structure

```
rag_based/
├── data/                              # PDF and generated index files
│   └── modern+database+10th+edition.pdf
├── scripts/                           # One-time data processing scripts
│   ├── chunk_pdf.py                   # Step 1: PDF → text chunks
│   ├── generate_embeddings.py         # Step 2: Chunks → embeddings
│   └── build_faiss_index.py           # Step 3: Embeddings → FAISS index
├── config.py                          # All configuration in one place
├── rag.py                             # RAG pipeline (search + LLM)
├── app.py                             # Streamlit web interface
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

1. Create virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and add your Gemini API key:

```bash
cp .env.example .env
```

3. Place the PDF in `data/` folder (if not already there).

4. Run the data processing pipeline (one time only):

```bash
python scripts/chunk_pdf.py
python scripts/generate_embeddings.py
python scripts/build_faiss_index.py
```

5. Start the application:

```bash
streamlit run app.py
```

## Tech Stack

- **Streamlit** — Web interface
- **PyPDF** — PDF text extraction
- **Sentence Transformers** (all-MiniLM-L6-v2) — Text embeddings
- **FAISS** — Vector similarity search
- **Google Gemini** — LLM answer generation

## Pipeline

```
PDF → Chunks → Embeddings → FAISS Index
                                  ↓
User Query → Embed → Search → Rank → Relevance Check → Gemini LLM → Answer
```
