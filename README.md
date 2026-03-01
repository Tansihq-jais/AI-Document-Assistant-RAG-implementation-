# AI Document Assistant 🤖📄

> **RAG-based document Q&A system** — Upload PDFs, ask questions, get cited answers.

Built with **FAISS** vector search, **HuggingFace** sentence-transformers, **OpenAI** GPT, **FastAPI** REST backend, and **Streamlit** frontend.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│              (frontend/app.py — port 8501)              │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP REST
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                       │
│              (backend/main.py — port 8000)              │
│                                                         │
│  ┌─────────────────┐    ┌──────────────────────────┐   │
│  │  Doc Processor  │    │      RAG Pipeline        │   │
│  │  (PyMuPDF +     │    │  retrieve → augment →    │   │
│  │   chunking)     │    │     generate (OpenAI)    │   │
│  └────────┬────────┘    └──────────────┬───────────┘   │
│           │                            │                │
│  ┌────────▼────────────────────────────▼───────────┐   │
│  │              Vector Store (FAISS)               │   │
│  │    sentence-transformers embeddings             │   │
│  │    Persisted to ./index_store/                  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Features

- **PDF Ingestion** — Upload any PDF; text is extracted page-by-page, cleaned, and chunked
- **Semantic Search** — HuggingFace `all-MiniLM-L6-v2` embeddings stored in FAISS
- **Cited Answers** — GPT generates answers citing exact source pages and files
- **Multi-document** — Index many PDFs, query all or filter to a specific document
- **Persistent Index** — FAISS index survives server restarts
- **REST API** — Full FastAPI backend with Swagger docs at `/docs`
- **Interactive UI** — Streamlit chat interface with real-time document management

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo>
cd rag-document-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Start the backend

```bash
cd backend
python main.py
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 4. Start the frontend

```bash
cd frontend
streamlit run app.py
# UI available at http://localhost:8501
```

---

## Project Structure

```
rag-document-assistant/
├── backend/
│   ├── main.py                 # FastAPI app, all REST endpoints
│   ├── rag_pipeline.py         # RAG orchestration (retrieve + generate)
│   ├── document_processor.py   # PDF parsing and text chunking
│   ├── vector_store.py         # FAISS index + HuggingFace embeddings
│   └── models.py               # Pydantic request/response schemas
├── frontend/
│   └── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System status + stats |
| `POST` | `/upload` | Upload & index a PDF |
| `GET` | `/documents` | List all indexed docs |
| `DELETE` | `/documents/{id}` | Remove a document |
| `POST` | `/query` | Ask a question |

Full interactive docs: **http://localhost:8000/docs**

### Example: Query via curl

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@research_paper.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main conclusions?", "top_k": 5}'
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | LLM model (`gpt-4`, `gpt-4o`, etc.) |
| `TEMPERATURE` | `0.2` | Generation temperature |
| `TOP_K` | `5` | Chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `INDEX_DIR` | `./index_store` | FAISS persistence directory |

---

## Customization

### Swap embedding model
```python
# In .env - higher accuracy, slower:
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Or BGE for state-of-the-art retrieval:
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Use a local LLM (Ollama)
Replace the OpenAI client in `rag_pipeline.py`:
```python
from openai import OpenAI
self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

### Tune chunking
Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env` for your document types:
- **Technical manuals**: larger chunks (700–1000 words)
- **Research papers**: medium chunks (400–600 words)  
- **Legal documents**: smaller chunks (200–400 words) with higher overlap

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF parsing | PyMuPDF (fitz) |
| Embeddings | HuggingFace sentence-transformers |
| Vector DB | FAISS (Facebook AI Similarity Search) |
| LLM | OpenAI GPT-3.5/4 |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Validation | Pydantic v2 |

---

## License

MIT
