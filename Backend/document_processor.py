"""
Document Processor
Handles PDF ingestion, text extraction, cleaning, and chunking.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk of text from a document."""
    chunk_id: str
    doc_id: str
    filename: str
    page_num: int
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_num": self.page_num,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "metadata": self.metadata,
        }


class DocumentProcessor:
    """
    Processes PDF documents into chunks suitable for embedding.

    Strategy:
    - Extract text page by page
    - Clean and normalize text
    - Chunk with overlap for context preservation
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_length: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def process_pdf(self, file_path: str, doc_id: str) -> List[DocumentChunk]:
        """Extract and chunk text from a PDF file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        filename = path.name
        logger.info(f"Processing PDF: {filename}")

        # Extract text from all pages
        pages_text = self._extract_pages(file_path)

        # Chunk text
        chunks = []
        chunk_index = 0

        for page_num, page_text in pages_text:
            cleaned = self._clean_text(page_text)
            if not cleaned.strip():
                continue

            page_chunks = self._chunk_text(cleaned)

            for chunk_text in page_chunks:
                if len(chunk_text.strip()) < self.min_chunk_length:
                    continue

                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index}",
                    doc_id=doc_id,
                    filename=filename,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    text=chunk_text.strip(),
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "total_pages": len(pages_text),
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

        logger.info(f"Extracted {len(chunks)} chunks from {filename}")
        return chunks

    def _extract_pages(self, file_path: str) -> List[tuple]:
        """Extract text from each page of a PDF."""
        pages = []
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                pages.append((page_num, text))
            doc.close()
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise
        return pages

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable chars
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        # Normalize newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)

            if end == len(words):
                break
            start += self.chunk_size - self.chunk_overlap

        return chunks
