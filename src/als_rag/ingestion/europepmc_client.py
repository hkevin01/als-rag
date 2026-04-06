"""Europe PMC REST API client for ALS literature ingestion."""

import logging
import time
from typing import List, Dict, Any
import requests

logger = logging.getLogger(__name__)

EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

ALS_EUROPEPMC_QUERIES = [
    "amyotrophic lateral sclerosis",
    "ALS TDP-43 aggregation",
    "SOD1 ALS familial",
    "C9orf72 hexanucleotide repeat ALS",
    "neurofilament light chain ALS biomarker",
    "ALSFRS-R clinical outcome ALS",
    "ALS frontotemporal dementia cognitive",
    "motor neuron disease respiratory failure",
    "ALS gene therapy antisense",
    "ALS neuroinflammation microglia",
    "ALS stem cell iPSC",
    "ALS survival prognosis",
]


class EuropePMCClient:
    """
    Fetch ALS literature from the Europe PMC REST API.

    Europe PMC covers 40M+ biomedical articles including PubMed,
    preprints, patents, and clinical guidelines. No API key required.

    Reference: https://europepmc.org/RestfulWebService
    """

    def __init__(self, rate_limit: float = 0.5):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "als-rag-research/0.1 (research use)"
        self.rate_limit = rate_limit
        self._last = 0.0

    def _wait(self):
        elapsed = time.time() - self._last
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last = time.time()

    def search(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search Europe PMC for a query.

        Args:
            query: Free-text search query.
            max_results: Maximum articles to return per query.

        Returns:
            List of article dicts with title, abstract, year, authors, url.
        """
        articles: List[Dict[str, Any]] = []
        cursor_mark = "*"
        page_size = min(100, max_results)

        params: Dict[str, Any] = {
            "query": query,
            "resultType": "core",
            "pageSize": page_size,
            "format": "json",
            "sort": "CITED desc",
            "cursorMark": cursor_mark,
        }

        while len(articles) < max_results:
            params["cursorMark"] = cursor_mark
            self._wait()
            try:
                resp = self.session.get(EUROPEPMC_API, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"Europe PMC search failed for '{query}': {e}")
                break

            result_list = data.get("resultList", {}).get("result", [])
            if not result_list:
                break

            for paper in result_list:
                abstract = paper.get("abstractText", "") or ""
                title = paper.get("title", "") or ""
                if not abstract and not title:
                    continue

                pmid = paper.get("pmid", "")
                doi = paper.get("doi", "")
                pub_year = str(paper.get("pubYear", ""))
                authors_list = paper.get("authorList", {}).get("author", [])
                authors = [
                    f"{a.get('lastName', '')} {a.get('initials', '')}".strip()
                    for a in (authors_list or [])
                ][:8]
                journal = paper.get("journalTitle", "") or ""
                source = paper.get("source", "")

                if pmid:
                    url = f"https://europepmc.org/article/MED/{pmid}"
                elif doi:
                    url = f"https://doi.org/{doi}"
                else:
                    url = f"https://europepmc.org/search?query={query}"

                articles.append({
                    "title": title,
                    "abstract": abstract,
                    "year": pub_year,
                    "authors": authors,
                    "journal": journal,
                    "pmid": pmid,
                    "doi": doi,
                    "url": url,
                    "source": f"europepmc_{source}",
                })

                if len(articles) >= max_results:
                    break

            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor

        return articles

    def fetch_als_corpus(self, papers_per_query: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch ALS corpus from Europe PMC across all ALS-focused queries.

        Returns:
            Deduplicated list of article dicts for the ingestion pipeline.
        """
        all_articles: List[Dict[str, Any]] = []
        seen: set = set()

        for query in ALS_EUROPEPMC_QUERIES:
            logger.info(f"Europe PMC: searching '{query}'")
            articles = self.search(query, max_results=papers_per_query)
            for a in articles:
                key = a.get("title", "")[:80]
                if key and key not in seen:
                    seen.add(key)
                    all_articles.append(a)

        logger.info(f"Europe PMC: fetched {len(all_articles)} unique articles")
        return all_articles
