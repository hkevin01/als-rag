"""FAISS vector store for ALS-RAG."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ALSVectorDB:
    """Thin wrapper around a FAISS IndexFlatIP + JSON metadata store."""

    def __init__(self, config=None):
        if config is None:
            from als_rag.utils.config import get_config
            config = get_config()
        self.config = config
        self._index = None
        self._metadata: List[Dict[str, Any]] = []

    def _load(self):
        import faiss
        if self.config.faiss_index_path.exists():
            self._index = faiss.read_index(str(self.config.faiss_index_path))
            with open(self.config.faiss_metadata_path) as f:
                self._metadata = json.load(f)
            logger.info(f"Loaded index with {self._index.ntotal} vectors")
        else:
            self._index = faiss.IndexFlatIP(self.config.embedding_dim)
            self._metadata = []

    def _ensure_loaded(self):
        if self._index is None:
            self._load()

    @property
    def total_vectors(self) -> int:
        self._ensure_loaded()
        return self._index.ntotal

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        if self._index.ntotal == 0:
            return []
        q = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        distances, indices = self._index.search(q, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = dict(self._metadata[idx])
            meta["score"] = float(score)
            results.append(meta)
        return results

    def reload(self):
        self._index = None
        self._load()
