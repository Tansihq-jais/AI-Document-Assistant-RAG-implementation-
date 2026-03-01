"""
AI Document Assistant - FastAPI Backend
REST API for document ingestion, management, and RAG-based querying.

Endpoints:
  POST /upload          - Upload and index a PDF
  GET  /documents       - List all indexed documents
  DELETE /documents/{id} - Remove a document
  POST /query           - Ask a question
  GET  /health          - Health check
"""

import os
import uuid
import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from models import (
    QueryRequest, QueryResponse, CitationResponse,
    DocumentListResponse, DocumentInfo,
    UploadResponse, DeleteResponse, HealthResponse,
)

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App State ─────────────────────────────────────────────────────────────────

class AppState:
    vector_store: VectorStore = None
    rag_pipeline: RAGPipeline = None
    doc_processor: DocumentProcessor = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    logger.info("Starting AI Document Assistant...")

    state.doc_processor = DocumentProcessor(
        chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
    )

    state.vector_store = VectorStore(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        index_dir=os.getenv("INDEX_DIR", "./index_store"),
    )

    state.rag_pipeline = RAGPipeline(
        vector_store=state.vector_store,
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        top_k=int(os.getenv("TOP_K", 5)),
        temperature=float(os.getenv("TEMPERATURE", 0.2)),
    )

    logger.info("✅ All components initialized")
    yield
    logger.info("Shutting down...")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Document Assistant",
    description="RAG-based document Q&A system with FAISS vector search and OpenAI generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_state() -> AppState:
    if state.vector_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return state


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(app_state: AppState = Depends(get_state)):
    """Check API health and indexed document stats."""
    return HealthResponse(
        status="healthy",
        total_documents=app_state.vector_store.total_documents,
        total_chunks=app_state.vector_store.total_chunks,
        model=app_state.rag_pipeline.model,
        embedding_model=app_state.vector_store.model_name,
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload and index"),
    app_state: AppState = Depends(get_state),
):
    """
    Upload a PDF document and index it for Q&A.

    The file will be processed, chunked, embedded with sentence-transformers,
    and stored in the FAISS index for semantic retrieval.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_size = 0
    doc_id = str(uuid.uuid4())

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        file_size = len(content)

        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")

        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Process and index
        chunks = app_state.doc_processor.process_pdf(tmp_path, doc_id)

        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted from the PDF")

        indexed = app_state.vector_store.add_documents(chunks)

        return UploadResponse(
            success=True,
            doc_id=doc_id,
            filename=file.filename,
            chunks_indexed=indexed,
            message=f"Successfully indexed {indexed} chunks from '{file.filename}'",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents(app_state: AppState = Depends(get_state)):
    """List all documents currently indexed."""
    docs = app_state.vector_store.get_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total_documents=app_state.vector_store.total_documents,
        total_chunks=app_state.vector_store.total_chunks,
    )


@app.delete("/documents/{doc_id}", response_model=dict, tags=["Documents"])
async def delete_document(
    doc_id: str,
    app_state: AppState = Depends(get_state),
):
    """Remove a document and all its chunks from the index."""
    removed = app_state.vector_store.remove_document(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    return {
        "success": True,
        "doc_id": doc_id,
        "chunks_removed": removed,
        "message": f"Removed {removed} chunks for document {doc_id}",
    }


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(
    request: QueryRequest,
    app_state: AppState = Depends(get_state),
):
    """
    Ask a question about your documents.

    The system will:
    1. Embed your question using sentence-transformers
    2. Retrieve the most semantically relevant chunks via FAISS
    3. Send context + question to the LLM for answer generation
    4. Return the answer with source citations
    """
    if app_state.vector_store.total_chunks == 0:
        raise HTTPException(
            status_code=422,
            detail="No documents indexed. Please upload a PDF first.",
        )

    # Optionally override model
    original_model = app_state.rag_pipeline.model
    if request.model:
        app_state.rag_pipeline.update_model(request.model)

    try:
        result = app_state.rag_pipeline.query(
            question=request.question,
            doc_id_filter=request.doc_id,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    finally:
        if request.model:
            app_state.rag_pipeline.update_model(original_model)

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                chunk_id=c.chunk_id,
                filename=c.filename,
                page_num=c.page_num,
                relevance_score=c.relevance_score,
                excerpt=c.excerpt,
            )
            for c in result.citations
        ],
        query=result.query,
        model_used=result.model_used,
        retrieved_chunks=result.retrieved_chunks,
        doc_filter=result.doc_filter,
    )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
    )
