"""
API Models
Pydantic schemas for FastAPI request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Question to ask")
    doc_id: Optional[str] = Field(None, description="Restrict search to a specific document")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    model: Optional[str] = Field(None, description="Override LLM model (e.g. gpt-4)")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main findings of this research?",
                "doc_id": None,
                "top_k": 5,
            }
        }


class DeleteDocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID to delete from the index")


# ── Response Models ───────────────────────────────────────────────────────────

class CitationResponse(BaseModel):
    chunk_id: str
    filename: str
    page_num: int
    relevance_score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]
    query: str
    model_used: str
    retrieved_chunks: int
    doc_filter: Optional[str] = None


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    total_chunks: int
    total_pages: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class UploadResponse(BaseModel):
    success: bool
    doc_id: str
    filename: str
    chunks_indexed: int
    message: str


class DeleteResponse(BaseModel):
    success: bool
    doc_id: str
    chunks_removed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    model: str
    embedding_model: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
