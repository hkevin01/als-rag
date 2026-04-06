"""Hybrid retriever: dense FAISS + BM25Okapi fusion + ALS query expansion.

Integration:
    rank_bm25 (dorianbrown/rank_bm25, Apache-2.0, 1.3k stars)
    BM25Okapi provides lexical matching complementary to dense cosine search.
    Final score = dense_weight * cosine_score + keyword_weight * bm25_norm_score
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines dense FAISS cosine search with BM25Okapi lexical ranking
    and ALS synonym query expansion for maximum recall on ALS literature.

    Score fusion:
        hybrid_score = dense_weight * cosine + keyword_weight * bm25_normalized

    Args:
        dense_weight: Weight for dense cosine score (0–1). Default 0.7.
    """

    def __init__(self, config=None, dense_weight: float = 0.7):
        if config is None:
            from als_rag.utils.config import get_config
            config = get_config()
        self.config = config
        self.dense_weight = dense_weight
        self.keyword_weight = 1.0 - dense_weight
        self._dense = None
        self._expander = None
        self._bm25 = None
        self._bm25_corpus: List[Dict[str, Any]] = []

    def _get_dense(self):
        if self._dense is None:
            from als_rag.retrieval.dense_retriever import DenseRetriever
            self._dense = DenseRetriever(self.config)
        return self._dense

    def _get_expander(self):
        if self._expander is None:
            from als_rag.retrieval.query_expander import ALSQueryExpander
            self._expander = ALSQueryExpander()
        return self._expander

    def _build_bm25(self, corpus: List[Dict[str, Any]]):
        """Build a BM25Okapi index from a list of metadata dicts."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed; BM25 disabled. Run: uv pip install rank-bm25")
            return

        tokenized = [
            (m.get("chunk_text", "") + " " + m.get("title", "")).lower().split()
            for m in corpus
        ]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus = corpus
        logger.info(f"BM25 index built over {len(corpus)} chunks")

    def _ensure_bm25(self):
        """Lazy-load the BM25 index from the FAISS metadata store."""
        if self._bm25 is not None:
            return
        try:
            import json
            meta_path = self.config.faiss_metadata_path
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
                if metadata:
                    self._build_bm25(metadata)
        except Exception as e:
            logger.warning(f"BM25 index build failed: {e}")

    def _bm25_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Return BM25-ranked candidates with a normalised 0–1 score."""
        self._ensure_bm25()
        if self._bm25 is None or not self._bm25_corpus:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        max_score = float(scores.max()) if scores.size > 0 else 1.0
        if max_score == 0:
            return []

        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            meta = dict(self._bm25_corpus[idx])
            meta["bm25_score"] = float(scores[idx]) / max_score
            results.append(meta)
        return results

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k passages using hybrid dense + BM25 fusion with
        ALS synonym query expansion.

        Returns chunks sorted by hybrid_score descending.
        """
        expander = self._get_expander()
        dense = self._get_dense()

        expanded_queries = expander.expand(query, max_expansions=2)
        all_results: Dict[str, Dict[str, Any]] = {}

        # --- Dense retrieval across expanded queries ---
        for i, q in enumerate(expanded_queries):
            weight_multiplier = 1.0 if i == 0 else 0.7
            results = dense.retrieve(q, top_k=top_k)
            for r in results:
                key = r.get("doc_id", r.get("title", ""))[:60]
                dense_score = r["score"] * self.dense_weight * weight_multiplier
                if key not in all_results:
                    all_results[key] = dict(r)
                    all_results[key]["dense_score"] = dense_score
                    all_results[key]["score"] = dense_score
                else:
                    if dense_score > all_results[key].get("dense_score", 0):
                        all_results[key]["dense_score"] = dense_score
                        all_results[key]["score"] = dense_score

        # --- BM25 retrieval on primary query ---
        bm25_results = self._bm25_retrieve(query, top_k=top_k * 2)
        for r in bm25_results:
            key = r.get("doc_id", r.get("title", ""))[:60]
            bm25_contribution = r["bm25_score"] * self.keyword_weight
            if key in all_results:
                all_results[key]["score"] = (
                    all_results[key].get("dense_score", 0) + bm25_contribution
                )
            else:
                all_results[key] = dict(r)
                all_results[key]["score"] = bm25_contribution

        ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
