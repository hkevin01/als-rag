"""PubMed client for fetching ALS research literature."""

import time
import logging
import requests
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ALS_SEARCH_TERMS = [
    "amyotrophic lateral sclerosis",
    "ALS motor neuron disease",
    "SOD1 ALS", "C9orf72 ALS", "TDP-43 ALS", "FUS ALS",
    "neurofilament ALS biomarker",
    "riluzole clinical trial",
    "edaravone ALS",
    "tofersen antisense ALS",
    "AMX0035 ALS",
    "ALSFRS-R outcome",
    "ALS FTD frontotemporal",
    "motor neuron disease survival",
    "ALS gene therapy",
    "ALS stem cell",
    "ALS respiratory failure",
    "ALS multidisciplinary care",
]

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_ESUM    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class PubMedClient:
    """Fetch ALS papers from PubMed."""

    def __init__(self, api_key: str = "", email: str = "", rate_limit: float = 0.34):
        self.api_key = api_key
        self.email = email
        self.rate_limit = 0.1 if api_key else rate_limit
        self._last_request = 0.0

    def _wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(self, query: str, max_results: int = 100) -> List[str]:
        """Return list of PubMed IDs for query."""
        self._wait()
        params: Dict[str, Any] = {
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json", "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        resp = requests.get(PUBMED_ESEARCH, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"PubMed search '{query[:50]}' → {len(pmids)} results")
        return pmids

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch title + abstract for a list of PMIDs."""
        if not pmids:
            return []
        self._wait()
        params: Dict[str, Any] = {
            "db": "pubmed", "id": ",".join(pmids),
            "rettype": "abstract", "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        resp = requests.get(PUBMED_EFETCH, params=params, timeout=60)
        resp.raise_for_status()
        return self._parse_xml(resp.text)

    def _parse_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(xml_text, "lxml-xml")
        articles = []
        for article in soup.find_all("PubmedArticle"):
            try:
                pmid = article.find("PMID").get_text(strip=True)
                title = article.find("ArticleTitle")
                title = title.get_text(strip=True) if title else ""
                abstract_parts = article.find_all("AbstractText")
                abstract = " ".join(p.get_text(strip=True) for p in abstract_parts)
                year_tag = article.find("PubDate")
                year = year_tag.find("Year").get_text(strip=True) if year_tag and year_tag.find("Year") else ""
                authors = [
                    f"{a.find('LastName', recursive=False).get_text(strip=True) if a.find('LastName') else ''} "
                    f"{a.find('Initials', recursive=False).get_text(strip=True) if a.find('Initials') else ''}"
                    for a in article.find_all("Author")
                ][:6]
                journal = article.find("Title")
                journal = journal.get_text(strip=True) if journal else ""
                articles.append({
                    "pmid": pmid, "title": title, "abstract": abstract,
                    "year": year, "authors": authors, "journal": journal,
                    "source": "pubmed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
            except Exception as exc:
                logger.warning(f"Failed to parse article: {exc}")
        return articles

    def fetch_als_corpus(self, max_per_term: int = 200) -> List[Dict[str, Any]]:
        """Fetch papers for all ALS search terms."""
        all_pmids: set = set()
        for term in ALS_SEARCH_TERMS:
            pmids = self.search(term, max_results=max_per_term)
            all_pmids.update(pmids)
            logger.info(f"Total unique PMIDs so far: {len(all_pmids)}")

        pmid_list = list(all_pmids)
        all_articles = []
        batch_size = 100
        for i in range(0, len(pmid_list), batch_size):
            batch = pmid_list[i : i + batch_size]
            articles = self.fetch_abstracts(batch)
            all_articles.extend(articles)
            logger.info(f"Fetched {len(all_articles)} / {len(pmid_list)} articles")

        return all_articles
