# AIOps — Agent AI Learning Journey

A progressive series of hands-on AI agent and RAG (Retrieval-Augmented Generation) projects built with Python, FastAPI, and various vector store backends.

---

## Structure

```
agent-ai/
├── day1/               # LLM basics & prompt engineering
├── day2/               # RAG with ChromaDB
├── day2_docling_lancedb/  # RAG with pymupdf + LanceDB + Ollama
└── day3/               # Conversational RAG with memory + Streamlit UI
```

---

## Projects

### Day 1 — LLM Fundamentals
Exploring direct LLM interaction with Groq.

**Stack:** Python, Groq API (`llama-3.1-8b-instant`), Jupyter notebooks

---

### Day 2 — RAG with ChromaDB
Upload PDFs and query them using semantic search backed by ChromaDB.

**Stack:** FastAPI, ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`), Groq, pypdf

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload and ingest a PDF |
| `POST` | `/query` | Ask a question against ingested docs |

---

### Day 2 (Docling + LanceDB) — Advanced RAG
PDF ingestion with pymupdf, vector storage in LanceDB, and local inference via Ollama.

**Stack:** FastAPI, pymupdf, LanceDB, sentence-transformers, Ollama (`llama3.2`) / Groq fallback

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload and ingest a PDF |
| `POST` | `/query` | Ask a question against ingested docs |

---

### Day 3 — Conversational RAG with Memory + Streamlit UI
Full-stack RAG app with conversation memory, a FastAPI backend, and a Streamlit chat UI.

**Stack:** FastAPI, ChromaDB, sentence-transformers, Groq, Streamlit, pypdf

**Run the backend:**
```bash
uvicorn app:app --reload
```

**Run the UI (in a separate terminal):**
```bash
streamlit run ui.py
```

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload and ingest a PDF |
| `POST` | `/query` | Ask a question with conversation memory |
| `GET` | `/documents` | List ingested documents |
| `GET` | `/memory` | View conversation history |
| `DELETE` | `/memory` | Clear conversation memory |

---

## Setup

### Prerequisites
- Python 3.10+
- [Groq API key](https://console.groq.com) (free tier)
- Ollama (optional, for day2_docling_lancedb local inference)

### Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r agent-ai/day3/requirements.txt
```

### Environment variables
Create a `.env` file in the relevant project folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Notes
- `chroma_db/`, `lancedb_data/`, `docs/`, and `.env` files are git-ignored
- Ollama must be running locally (`ollama serve`) for the LanceDB variant
