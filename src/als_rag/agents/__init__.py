"""
ALS-RAG Agent Layer.

Five autonomous agents that orchestrate the RAG pipeline for different
research workflows:

    ResearchAgent                — end-to-end Q&A with source citation
    IngestionAgent               — multi-source corpus refresh orchestrator
    ClinicalMatchingAgent        — case-based literature retrieval from clinical data
    SystematicReviewAgent        — structured evidence synthesis across a topic
    CitationVerificationAgent    — hallucination guard / citation coverage audit
"""

from als_rag.agents.research_agent import ResearchAgent
from als_rag.agents.ingestion_agent import IngestionAgent
from als_rag.agents.clinical_agent import ClinicalMatchingAgent
from als_rag.agents.review_agent import SystematicReviewAgent
from als_rag.agents.citation_agent import CitationVerificationAgent

__all__ = [
    "ResearchAgent",
    "IngestionAgent",
    "ClinicalMatchingAgent",
    "SystematicReviewAgent",
    "CitationVerificationAgent",
]
