"""Semantic Scholar client for ALS literature."""

import logging
import time
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

ALS_SCHOLAR_QUERIES = [
    "amyotrophic lateral sclerosis",
    "ALS TDP-43 pathology",
    "SOD1 ALS mutation",
    "C9orf72 hexanucleotide repeat ALS",
    "neurofilament light chain ALS biomarker",
    "ALSFRS-R clinical trial outcome",
    "riluzole motor neuron disease",
    "tofersen antisense oligonucleotide",
    "ALS gene therapy clinical",
    "ALS FTD cognitive impairment",
]

BASE_URL = "https://api.semanticscholar.org/graph/v1"


class ScholarClient:
    def __init__(self, api_key: Optional[str] = None):
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "paperId,title,abstract,year,authors,venue,externalIds,openAccessPdf",
        }
        try:
            resp = self.session.get(f"{BASE_URL}/paper/search", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed for '{query}': {e}")
            return []
        results = []
        for p in data.get("data", []):
            results.append({
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "year": str(p.get("year", "")),
                "authors": [a.get("name", "") for a in p.get("authors", [])],
                "journal": p.get("venue", ""),
                "doi": (p.get("externalIds") or {}).get("DOI", ""),
                "url": f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
                "source": "semantic_scholar",
            })
        return results

    def fetch_als_corpus(self, papers_per_query: int = 50) -> List[Dict[str, Any]]:
        all_articles: List[Dict[str, Any]] = []
        seen: set = set()
        for query in ALS_SCHOLAR_QUERIES:
            logger.info(f"Searching Semantic Scholar: {query}")
            articles = self.search(query, limit=papers_per_query)
            for a in articles:
                key = a.get("title", "")[:80]
                if key and key not in seen:
                    seen.add(key)
                    all_articles.append(a)
            time.sleep(1.0)
        logger.info(f"Semantic Scholar: fetched {len(all_articles)} unique articles")
        return all_articles
