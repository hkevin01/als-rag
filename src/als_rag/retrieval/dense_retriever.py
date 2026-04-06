"""Dense FAISS retriever for ALS-RAG."""

import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, config=None):
        if config is None:
            from als_rag.utils.config import get_config
            config = get_config()
        self.config = config
        self._embedder = None
        self._db = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    def _get_db(self):
        if self._db is None:
            from als_rag.storage.vector_db import ALSVectorDB
            self._db = ALSVectorDB(self.config)
        return self._db

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        embedder = self._get_embedder()
        db = self._get_db()
        embedding = embedder.encode([query], convert_to_numpy=True)[0]
        results = db.search(embedding, top_k=top_k)
        logger.debug(f"Dense retrieval for '{query[:50]}': {len(results)} results")
        return results
