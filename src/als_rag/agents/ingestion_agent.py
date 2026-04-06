"""
IngestionAgent — multi-source corpus refresh orchestrator.

Coordinates fetching from PubMed, Semantic Scholar, arXiv,
ClinicalTrials.gov, and Europe PMC, then indexes into FAISS.
Provides progress callbacks and a structured IngestionReport.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    """Summary of a completed ingestion run."""

    pubmed_count: int = 0
    scholar_count: int = 0
    arxiv_count: int = 0
    clinicaltrials_count: int = 0
    europepmc_count: int = 0
    total_articles: int = 0
    total_chunks_indexed: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Ingestion Report ===",
            f"  PubMed:             {self.pubmed_count}",
            f"  Semantic Scholar:   {self.scholar_count}",
            f"  arXiv:              {self.arxiv_count}",
            f"  ClinicalTrials.gov: {self.clinicaltrials_count}",
            f"  Europe PMC:         {self.europepmc_count}",
            f"  ─────────────────────",
            f"  Total articles:     {self.total_articles}",
            f"  Chunks indexed:     {self.total_chunks_indexed}",
        ]
        if self.errors:
            lines.append(f"  Errors:             {len(self.errors)}")
        return "\n".join(lines)


class IngestionAgent:
    """
    Orchestrates a full ALS corpus refresh from all configured sources.

    Integrated sources:
        1. PubMed (NCBI E-utilities)
        2. Semantic Scholar Academic Graph API
        3. arXiv REST API
        4. ClinicalTrials.gov REST API v2  [NEW]
        5. Europe PMC REST API             [NEW]

    Usage:
        agent = IngestionAgent()
        report = agent.run()
        print(report.summary())

    With progress callback:
        def on_progress(msg): print(msg)
        report = agent.run(on_progress=on_progress)
    """

    def __init__(
        self,
        config=None,
        sources: Optional[List[str]] = None,
        max_per_source: int = 100,
    ):
        """
        Args:
            config: ALSConfig instance (auto-loaded if None).
            sources: List of source names to enable. Defaults to all.
                     Options: ["pubmed", "scholar", "arxiv", "clinicaltrials", "europepmc"]
            max_per_source: Max articles per source (per query bucket).
        """
        if config is None:
            from als_rag.utils.config import get_config
            config = get_config()
        self.config = config
        self.sources = sources or ["pubmed", "scholar", "arxiv", "clinicaltrials", "europepmc"]
        self.max_per_source = max_per_source

    def run(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> IngestionReport:
        """
        Run the full ingestion pipeline across all enabled sources.

        Args:
            on_progress: Optional callback(message: str) for progress updates.

        Returns:
            IngestionReport with per-source article counts and total chunks.
        """
        def log(msg: str):
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        report = IngestionReport()
        all_articles: List[Dict[str, Any]] = []

        if "pubmed" in self.sources:
            try:
                from als_rag.ingestion.pubmed_client import PubMedClient
                log("Fetching from PubMed...")
                pm = PubMedClient(
                    api_key=os.environ.get("PUBMED_API_KEY", ""),
                    contact_email=os.environ.get("CONTACT_EMAIL", "user@example.com"),
                )
                articles = pm.fetch_als_corpus(max_per_query=self.max_per_source)
                report.pubmed_count = len(articles)
                all_articles.extend(articles)
                log(f"PubMed: {len(articles)} articles")
            except Exception as e:
                msg = f"PubMed ingestion error: {e}"
                logger.error(msg)
                report.errors.append(msg)

        if "scholar" in self.sources:
            try:
                from als_rag.ingestion.scholar_client import ScholarClient
                log("Fetching from Semantic Scholar...")
                sc = ScholarClient(api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
                articles = sc.fetch_als_corpus(papers_per_query=self.max_per_source)
                report.scholar_count = len(articles)
                all_articles.extend(articles)
                log(f"Semantic Scholar: {len(articles)} articles")
            except Exception as e:
                msg = f"Semantic Scholar ingestion error: {e}"
                logger.error(msg)
                report.errors.append(msg)

        if "arxiv" in self.sources:
            try:
                from als_rag.ingestion.arxiv_client import ArxivClient
                log("Fetching from arXiv...")
                ax = ArxivClient()
                articles = ax.fetch_als_corpus(results_per_query=self.max_per_source)
                report.arxiv_count = len(articles)
                all_articles.extend(articles)
                log(f"arXiv: {len(articles)} articles")
            except Exception as e:
                msg = f"arXiv ingestion error: {e}"
                logger.error(msg)
                report.errors.append(msg)

        if "clinicaltrials" in self.sources:
            try:
                from als_rag.ingestion.clinicaltrials_client import ClinicalTrialsClient
                log("Fetching from ClinicalTrials.gov...")
                ct = ClinicalTrialsClient()
                articles = ct.fetch_als_corpus(max_results=self.max_per_source * 2)
                report.clinicaltrials_count = len(articles)
                all_articles.extend(articles)
                log(f"ClinicalTrials.gov: {len(articles)} trials")
            except Exception as e:
                msg = f"ClinicalTrials.gov ingestion error: {e}"
                logger.error(msg)
                report.errors.append(msg)

        if "europepmc" in self.sources:
            try:
                from als_rag.ingestion.europepmc_client import EuropePMCClient
                log("Fetching from Europe PMC...")
                epmc = EuropePMCClient()
                articles = epmc.fetch_als_corpus(papers_per_query=self.max_per_source)
                report.europepmc_count = len(articles)
                all_articles.extend(articles)
                log(f"Europe PMC: {len(articles)} articles")
            except Exception as e:
                msg = f"Europe PMC ingestion error: {e}"
                logger.error(msg)
                report.errors.append(msg)

        # Deduplicate by title prefix
        seen: set = set()
        unique_articles: List[Dict[str, Any]] = []
        for a in all_articles:
            key = a.get("title", "")[:80]
            if key and key not in seen:
                seen.add(key)
                unique_articles.append(a)

        report.total_articles = len(unique_articles)
        log(f"Total unique articles: {report.total_articles} — indexing...")

        if unique_articles:
            from als_rag.ingestion.pipeline import ALSIngestionPipeline
            pipeline = ALSIngestionPipeline(self.config)
            report.total_chunks_indexed = pipeline.ingest(unique_articles)
            log(f"Indexed {report.total_chunks_indexed} chunks into FAISS")

        log(report.summary())
        return report
