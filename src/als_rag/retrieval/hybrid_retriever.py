"""Hybrid retriever: dense + BM25 keyword fusion + query expansion."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines dense vector search with keyword-based BM25 and query expansion."""

    def __init__(self, config=None, dense_weight: float = 0.7):
        if config is None:
            from als_rag.utils.config import get_config
            config = get_config()
        self.config = config
        self.dense_weight = dense_weight
        self.keyword_weight = 1.0 - dense_weight
        self._dense = None
        self._expander = None

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

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        expander = self._get_expander()
        dense = self._get_dense()

        expanded_queries = expander.expand(query, max_expansions=2)
        all_results: Dict[str, Dict[str, Any]] = {}

        for i, q in enumerate(expanded_queries):
            weight_multiplier = 1.0 if i == 0 else 0.7
            results = dense.retrieve(q, top_k=top_k)
            for r in results:
                key = r.get("doc_id", r.get("title", ""))[:60]
                if key not in all_results:
                    all_results[key] = dict(r)
                    all_results[key]["score"] = r["score"] * self.dense_weight * weight_multiplier
                else:
                    all_results[key]["score"] = max(
                        all_results[key]["score"],
                        r["score"] * self.dense_weight * weight_multiplier
                    )

        ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
