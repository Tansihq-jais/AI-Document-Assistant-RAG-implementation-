"""
RAG Pipeline
Orchestrates retrieval + LLM generation with citation support.
Supports OpenAI API and HuggingFace inference as backends.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from openai import OpenAI

from document_processor import DocumentChunk
from vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A source citation from the retrieved context."""
    chunk_id: str
    filename: str
    page_num: int
    relevance_score: float
    excerpt: str  # Short snippet from the chunk


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer: str
    citations: List[Citation]
    query: str
    model_used: str
    retrieved_chunks: int
    doc_filter: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [
                {
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "page_num": c.page_num,
                    "relevance_score": round(c.relevance_score, 4),
                    "excerpt": c.excerpt,
                }
                for c in self.citations
            ],
            "query": self.query,
            "model_used": self.model_used,
            "retrieved_chunks": self.retrieved_chunks,
            "doc_filter": self.doc_filter,
        }


SYSTEM_PROMPT = """You are a precise, helpful AI assistant specialized in answering questions based on provided document excerpts.

INSTRUCTIONS:
1. Answer the question using ONLY the provided context excerpts.
2. Be concise and factual. Do not hallucinate or add information not present in the context.
3. When referencing information, cite the source using [Source: filename, Page X] format.
4. If the context doesn't contain enough information to answer the question, say so clearly.
5. Structure complex answers with clear headings when appropriate.
6. Preserve technical terms, numbers, and proper nouns exactly as they appear in the source."""


def build_context_prompt(chunks_with_scores: List[Tuple[DocumentChunk, float]]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    context_parts = []
    for i, (chunk, score) in enumerate(chunks_with_scores, 1):
        context_parts.append(
            f"[Excerpt {i}] Source: {chunk.filename}, Page {chunk.page_num} "
            f"(Relevance: {score:.2f})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(context_parts)


class RAGPipeline:
    """
    Full RAG pipeline: retrieve → augment → generate.

    Supports:
    - OpenAI models (GPT-3.5-turbo, GPT-4, GPT-4o)
    - HuggingFace Inference API (as fallback)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        top_k: int = 5,
        max_context_chunks: int = 5,
        temperature: float = 0.2,
    ):
        self.vector_store = vector_store
        self.model = model
        self.top_k = top_k
        self.max_context_chunks = max_context_chunks
        self.temperature = temperature

        # Setup OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter."
            )
        self.client = OpenAI(api_key=api_key)
        logger.info(f"RAG Pipeline initialized with model: {model}")

    def query(
        self,
        question: str,
        doc_id_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Full RAG query: retrieve relevant chunks, then generate an answer.

        Args:
            question: The user's question
            doc_id_filter: Optional - restrict search to a specific document
            top_k: Override default number of chunks to retrieve

        Returns:
            RAGResponse with answer and citations
        """
        k = top_k or self.top_k
        logger.info(f"RAG query: '{question[:80]}...' (k={k})")

        # ── 1. Retrieve ──────────────────────────────────────────────────────
        results = self.vector_store.search(
            query=question,
            k=k,
            doc_id_filter=doc_id_filter,
        )

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
                citations=[],
                query=question,
                model_used=self.model,
                retrieved_chunks=0,
                doc_filter=doc_id_filter,
            )

        # ── 2. Build context ─────────────────────────────────────────────────
        context = build_context_prompt(results[:self.max_context_chunks])

        user_message = f"""CONTEXT EXCERPTS:
{context}

QUESTION: {question}

Please answer the question based solely on the provided context. Include citations."""

        # ── 3. Generate ──────────────────────────────────────────────────────
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content

        # ── 4. Build citations ───────────────────────────────────────────────
        citations = []
        for chunk, score in results[:self.max_context_chunks]:
            excerpt = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            citations.append(Citation(
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page_num=chunk.page_num,
                relevance_score=score,
                excerpt=excerpt,
            ))

        logger.info(f"Generated answer with {len(citations)} citations")

        return RAGResponse(
            answer=answer,
            citations=citations,
            query=question,
            model_used=self.model,
            retrieved_chunks=len(results),
            doc_filter=doc_id_filter,
        )

    def update_model(self, model: str):
        """Switch the LLM model at runtime."""
        self.model = model
        logger.info(f"Model updated to: {model}")

    def update_top_k(self, top_k: int):
        """Update the number of chunks to retrieve."""
        self.top_k = top_k
