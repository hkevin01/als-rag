"""arXiv client for ALS-related preprints."""

import logging
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import requests

logger = logging.getLogger(__name__)

ALS_ARXIV_QUERIES = [
    "amyotrophic lateral sclerosis machine learning",
    "ALS biomarker deep learning",
    "motor neuron disease NLP clinical notes",
    "ALS TDP-43 computational biology",
    "ALS survival prediction neural network",
    "C9orf72 repeat expansion bioinformatics",
    "ALS drug discovery artificial intelligence",
]

ARXIV_API = "https://export.arxiv.org/api/query"
NS = "http://www.w3.org/2005/Atom"


class ArxivClient:
    def search(self, query: str, max_results: int = 30) -> List[Dict[str, Any]]:
        params = {
            "search_query": f"all:{query}",
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            resp = requests.get(ARXIV_API, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
            return []
        root = ET.fromstring(resp.text)
        results = []
        for entry in root.findall(f"{{{NS}}}entry"):
            title = entry.findtext(f"{{{NS}}}title", "").replace("\n", " ").strip()
            abstract = entry.findtext(f"{{{NS}}}summary", "").replace("\n", " ").strip()
            arxiv_id = entry.findtext(f"{{{NS}}}id", "").strip()
            year = ""
            pub = entry.findtext(f"{{{NS}}}published", "")
            if pub:
                year = pub[:4]
            authors = [a.findtext(f"{{{NS}}}name", "") for a in entry.findall(f"{{{NS}}}author")]
            results.append({
                "title": title,
                "abstract": abstract,
                "year": year,
                "authors": authors,
                "journal": "arXiv",
                "url": arxiv_id,
                "source": "arxiv",
            })
        return results

    def fetch_als_corpus(self, results_per_query: int = 30) -> List[Dict[str, Any]]:
        all_articles: List[Dict[str, Any]] = []
        seen: set = set()
        for query in ALS_ARXIV_QUERIES:
            logger.info(f"Searching arXiv: {query}")
            articles = self.search(query, max_results=results_per_query)
            for a in articles:
                key = a.get("title", "")[:80]
                if key and key not in seen:
                    seen.add(key)
                    all_articles.append(a)
            time.sleep(3.0)  # arXiv rate limit
        logger.info(f"arXiv: fetched {len(all_articles)} unique articles")
        return all_articles
