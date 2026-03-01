"""
Vector Store
Manages FAISS index for fast semantic similarity search.
Uses HuggingFace sentence-transformers for embeddings.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from document_processor import DocumentChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("./index_store")


class VectorStore:
    """
    FAISS-backed vector store for semantic document search.

    - Embeds chunks using sentence-transformers
    - Stores vectors in FAISS flat L2 index (cosine via normalization)
    - Persists index + metadata to disk
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        index_dir: str = str(INDEX_DIR),
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []
        self.doc_ids: List[str] = []  # Track which doc each chunk belongs to

        self._load_or_create_index()

    # ── Index lifecycle ─────────────────────────────────────────────────────

    def _load_or_create_index(self):
        """Load existing index from disk or create a fresh one."""
        index_path = self.index_dir / "faiss.index"
        meta_path = self.index_dir / "metadata.pkl"

        if index_path.exists() and meta_path.exists():
            logger.info("Loading existing FAISS index from disk")
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.doc_ids = data.get("doc_ids", [c.doc_id for c in self.chunks])
        else:
            logger.info("Creating new FAISS index")
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine with normalization)

    def _save_index(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "metadata.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "doc_ids": self.doc_ids}, f)
        logger.info(f"Saved index with {len(self.chunks)} chunks")

    # ── Indexing ─────────────────────────────────────────────────────────────

    def add_documents(self, chunks: List[DocumentChunk]) -> int:
        """Embed and index a list of document chunks."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")

        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )
        embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.doc_ids.extend([c.doc_id for c in chunks])

        self._save_index()
        logger.info(f"Indexed {len(chunks)} chunks. Total: {len(self.chunks)}")
        return len(chunks)

    def remove_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document and rebuild index."""
        indices_to_keep = [i for i, did in enumerate(self.doc_ids) if did != doc_id]
        removed = len(self.chunks) - len(indices_to_keep)

        if removed == 0:
            return 0

        # Rebuild index without the document
        remaining_chunks = [self.chunks[i] for i in indices_to_keep]
        self.chunks = []
        self.doc_ids = []
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        if remaining_chunks:
            texts = [c.text for c in remaining_chunks]
            embeddings = self.encoder.encode(texts, normalize_embeddings=True).astype(np.float32)
            self.index.add(embeddings)
            self.chunks = remaining_chunks
            self.doc_ids = [c.doc_id for c in remaining_chunks]

        self._save_index()
        logger.info(f"Removed {removed} chunks for doc_id={doc_id}")
        return removed

    # ── Retrieval ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        doc_id_filter: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for the most relevant chunks for a query.

        Returns list of (chunk, similarity_score) tuples.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = self.encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # Retrieve more if filtering
        fetch_k = k * 3 if doc_id_filter else k
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if doc_id_filter and chunk.doc_id != doc_id_filter:
                continue
            results.append((chunk, float(score)))
            if len(results) == k:
                break

        return results

    # ── Metadata ─────────────────────────────────────────────────────────────

    def get_documents(self) -> List[Dict[str, Any]]:
        """Return summary of all indexed documents."""
        seen = {}
        for chunk in self.chunks:
            if chunk.doc_id not in seen:
                seen[chunk.doc_id] = {
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "total_chunks": 0,
                    "pages": set(),
                }
            seen[chunk.doc_id]["total_chunks"] += 1
            seen[chunk.doc_id]["pages"].add(chunk.page_num)

        result = []
        for doc in seen.values():
            doc["total_pages"] = len(doc["pages"])
            del doc["pages"]
            result.append(doc)
        return result

    def get_document_chunks(self, doc_id: str) -> List[DocumentChunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_documents(self) -> int:
        return len(set(self.doc_ids))
