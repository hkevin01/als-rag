"""ALS-RAG ingestion pipeline: fetch → chunk → embed → index."""

import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def doc_id(article: Dict[str, Any]) -> str:
    key = article.get("pmid") or article.get("doi") or article.get("title", "")
    return hashlib.md5(key.encode()).hexdigest()


class ALSIngestionPipeline:
    """End-to-end ingestion: articles → FAISS vector store."""

    def __init__(self, config=None):
        if config is None:
            from als_rag.utils.config import get_config
            config = get_config()
        self.config = config
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    def ingest(self, articles: List[Dict[str, Any]]) -> int:
        """Ingest articles into the vector store. Returns number of chunks added."""
        import numpy as np
        import faiss

        embedder = self._get_embedder()
        chunks: List[str] = []
        metadata: List[Dict[str, Any]] = []

        logger.info(f"Chunking {len(articles)} articles...")
        for article in tqdm(articles, desc="Chunking"):
            text = f"{article.get('title', '')} {article.get('abstract', '')}"
            for chunk in chunk_text(text, self.config.chunk_size, self.config.chunk_overlap):
                chunks.append(chunk)
                metadata.append({
                    "doc_id": doc_id(article),
                    "pmid": article.get("pmid", ""),
                    "title": article.get("title", ""),
                    "year": article.get("year", ""),
                    "authors": article.get("authors", []),
                    "journal": article.get("journal", ""),
                    "url": article.get("url", ""),
                    "chunk_text": chunk,
                })

        logger.info(f"Embedding {len(chunks)} chunks...")
        embeddings = embedder.encode(
            chunks, batch_size=self.config.batch_size,
            show_progress_bar=True, convert_to_numpy=True
        )
        embeddings = embeddings.astype(np.float32)

        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

        # Build / update FAISS index
        index_path = self.config.faiss_index_path
        meta_path = self.config.faiss_metadata_path

        if index_path.exists():
            index = faiss.read_index(str(index_path))
            with open(meta_path) as f:
                existing_meta = json.load(f)
        else:
            index = faiss.IndexFlatIP(self.config.embedding_dim)
            existing_meta = []

        index.add(embeddings)
        faiss.write_index(index, str(index_path))

        existing_meta.extend(metadata)
        with open(meta_path, "w") as f:
            json.dump(existing_meta, f)

        logger.info(f"Index now has {index.ntotal} vectors")
        return len(chunks)
