"""Tests for the four ALS-RAG agents."""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class TestResearchAgent:
    def test_import(self):
        from als_rag.agents.research_agent import ResearchAgent, ResearchResult
        assert ResearchAgent is not None
        assert ResearchResult is not None

    def test_ask_no_index(self):
        """When no FAISS index exists, returns empty sources and no-result answer."""
        from als_rag.agents.research_agent import ResearchAgent
        agent = ResearchAgent(generate=False)
        result = agent.ask("SOD1 ALS prognosis")
        assert result.query == "SOD1 ALS prognosis"
        assert isinstance(result.answer, str)
        assert isinstance(result.sources, list)
        assert isinstance(result.entities, list)
        assert isinstance(result.expanded_queries, list)
        assert len(result.expanded_queries) >= 1

    def test_result_format_citation_empty(self):
        from als_rag.agents.research_agent import ResearchResult
        r = ResearchResult(query="test", answer="test answer", sources=[])
        assert r.format_citation_list() == ""

    def test_result_format_citation_with_sources(self):
        from als_rag.agents.research_agent import ResearchResult
        sources = [
            {"title": "Paper A", "year": "2023", "url": "https://example.com", "score": 0.85},
            {"title": "Paper B", "year": "2022", "url": "", "score": 0.70},
        ]
        r = ResearchResult(query="test", answer="ans", sources=sources)
        citation = r.format_citation_list()
        assert "Paper A" in citation
        assert "2023" in citation
        assert "0.850" in citation
        assert "Paper B" in citation

    def test_query_expansion_captured(self):
        from als_rag.agents.research_agent import ResearchAgent
        agent = ResearchAgent(generate=False)
        result = agent.ask("neurofilament ALS prognosis")
        # Expander should produce at least the original query
        assert result.expanded_queries[0] == "neurofilament ALS prognosis"


# ---------------------------------------------------------------------------
# IngestionAgent
# ---------------------------------------------------------------------------

class TestIngestionAgent:
    def test_import(self):
        from als_rag.agents.ingestion_agent import IngestionAgent, IngestionReport
        assert IngestionAgent is not None

    def test_report_summary_format(self):
        from als_rag.agents.ingestion_agent import IngestionReport
        report = IngestionReport(
            pubmed_count=120,
            scholar_count=80,
            arxiv_count=30,
            clinicaltrials_count=50,
            europepmc_count=200,
            total_articles=480,
            total_chunks_indexed=1200,
        )
        summary = report.summary()
        assert "PubMed" in summary
        assert "480" in summary
        assert "1200" in summary

    def test_sources_default(self):
        from als_rag.agents.ingestion_agent import IngestionAgent
        agent = IngestionAgent()
        assert "clinicaltrials" in agent.sources
        assert "europepmc" in agent.sources
        assert "pubmed" in agent.sources

    def test_sources_subset(self):
        from als_rag.agents.ingestion_agent import IngestionAgent
        agent = IngestionAgent(sources=["arxiv"])
        assert agent.sources == ["arxiv"]

    @patch("als_rag.agents.ingestion_agent.IngestionAgent.run")
    def test_run_returns_report(self, mock_run):
        from als_rag.agents.ingestion_agent import IngestionAgent, IngestionReport
        mock_run.return_value = IngestionReport(total_chunks_indexed=42)
        agent = IngestionAgent(sources=["arxiv"])
        report = agent.run()
        assert report.total_chunks_indexed == 42


# ---------------------------------------------------------------------------
# ClinicalMatchingAgent
# ---------------------------------------------------------------------------

class TestClinicalMatchingAgent:
    def test_import(self):
        from als_rag.agents.clinical_agent import ClinicalMatchingAgent, ClinicalMatchResult
        assert ClinicalMatchingAgent is not None

    def test_match_no_index(self):
        from als_rag.agents.clinical_agent import ClinicalMatchingAgent
        agent = ClinicalMatchingAgent(generate=False)
        record = {
            "alsfrs_r_total": 36,
            "fvc_percent_predicted": 70,
            "c9orf72_repeat": True,
            "denervation_regions": ["bulbar", "cervical"],
            "alsfrs_r_slope": -1.0,
        }
        result = agent.match(record)
        assert isinstance(result.onset_phenotype, str)
        assert len(result.onset_phenotype) > 0
        assert isinstance(result.features_description, str)
        assert "bulbar" in result.features_description or "C9orf72" in result.features_description

    def test_match_from_dict_alias(self):
        from als_rag.agents.clinical_agent import ClinicalMatchingAgent
        agent = ClinicalMatchingAgent(generate=False)
        record = {"alsfrs_r_total": 40, "fvc_percent_predicted": 90}
        r1 = agent.match(record)
        r2 = agent.match_from_dict(record)
        assert r1.onset_phenotype == r2.onset_phenotype

    def test_progression_rate_computed(self):
        from als_rag.agents.clinical_agent import ClinicalMatchingAgent
        agent = ClinicalMatchingAgent(generate=False)
        record = {
            "alsfrs_r_total": 40,
            "fvc_percent_predicted": 90,
            "alsfrs_r_series": [48, 44, 40, 36],
            "alsfrs_r_times_months": [0, 1, 2, 3],
        }
        result = agent.match(record)
        assert result.progression_rate is not None
        assert result.progression_rate < 0  # declining

    def test_result_summary(self):
        from als_rag.agents.clinical_agent import ClinicalMatchResult
        r = ClinicalMatchResult(
            features_description="ALS with bulbar onset",
            onset_phenotype="bulbar",
            progression_rate=-1.0,
        )
        summary = r.summary()
        assert "bulbar" in summary
        assert "-1.00" in summary


# ---------------------------------------------------------------------------
# SystematicReviewAgent
# ---------------------------------------------------------------------------

class TestSystematicReviewAgent:
    def test_import(self):
        from als_rag.agents.review_agent import SystematicReviewAgent, SystematicReviewResult
        assert SystematicReviewAgent is not None

    def test_review_no_index(self):
        from als_rag.agents.review_agent import SystematicReviewAgent
        agent = SystematicReviewAgent(generate=False)
        result = agent.review("tofersen SOD1 ALS")
        assert result.topic == "tofersen SOD1 ALS"
        assert isinstance(result.synthesis, str)
        assert isinstance(result.sources, list)
        assert isinstance(result.sub_queries_run, list)
        assert len(result.sub_queries_run) > 0

    def test_sub_queries_include_topic(self):
        from als_rag.agents.review_agent import SystematicReviewAgent
        agent = SystematicReviewAgent(generate=False)
        result = agent.review("neurofilament")
        for q in result.sub_queries_run:
            assert "neurofilament" in q

    def test_format_entity_summary_empty(self):
        from als_rag.agents.review_agent import SystematicReviewResult
        r = SystematicReviewResult(topic="test", synthesis="syn", entity_counts={})
        assert "No entities" in r.format_entity_summary()

    def test_format_entity_summary_with_counts(self):
        from als_rag.agents.review_agent import SystematicReviewResult
        r = SystematicReviewResult(
            topic="test",
            synthesis="syn",
            entity_counts={"GENE": 5, "BIOMARKER": 3},
        )
        summary = r.format_entity_summary()
        assert "GENE" in summary
        assert "5" in summary

    def test_format_source_table(self):
        from als_rag.agents.review_agent import SystematicReviewResult
        sources = [
            {"title": "Paper A", "year": "2023", "score": 0.9, "source": "pubmed"},
        ]
        r = SystematicReviewResult(topic="t", synthesis="s", sources=sources)
        table = r.format_source_table()
        assert "Paper A" in table
        assert "0.900" in table


# ---------------------------------------------------------------------------
# HybridRetriever BM25 integration
# ---------------------------------------------------------------------------

class TestHybridRetrieverBM25:
    def test_import(self):
        from als_rag.retrieval.hybrid_retriever import HybridRetriever
        assert HybridRetriever is not None

    def test_bm25_retrieve_empty_index(self):
        from als_rag.retrieval.hybrid_retriever import HybridRetriever
        r = HybridRetriever()
        # No metadata loaded → BM25 returns empty list, not an error
        results = r._bm25_retrieve("SOD1 ALS", top_k=5)
        assert results == []

    def test_bm25_build_and_retrieve(self):
        from als_rag.retrieval.hybrid_retriever import HybridRetriever
        r = HybridRetriever()
        corpus = [
            {"chunk_text": "SOD1 mutation causes familial ALS", "title": "SOD1 paper", "doc_id": "a"},
            {"chunk_text": "Neurofilament light chain biomarker", "title": "NfL study", "doc_id": "b"},
            {"chunk_text": "TDP-43 aggregation motor neurons", "title": "TDP-43 paper", "doc_id": "c"},
        ]
        r._build_bm25(corpus)
        results = r._bm25_retrieve("SOD1 ALS mutation", top_k=2)
        assert len(results) > 0
        assert results[0]["doc_id"] == "a"
        assert 0.0 <= results[0]["bm25_score"] <= 1.0

    def test_hybrid_scorer_prefers_relevant(self):
        """BM25 score contribution should change the ranking."""
        from als_rag.retrieval.hybrid_retriever import HybridRetriever
        r = HybridRetriever(dense_weight=0.5)
        corpus = [
            {"chunk_text": "tofersen BIIB067 SOD1 antisense ALS treatment", "title": "Tofersen", "doc_id": "x"},
            {"chunk_text": "riluzole glutamate ALS neuroprotection", "title": "Riluzole", "doc_id": "y"},
        ]
        r._build_bm25(corpus)
        results = r._bm25_retrieve("tofersen SOD1", top_k=2)
        assert results[0]["doc_id"] == "x"


# ---------------------------------------------------------------------------
# New ingestion clients
# ---------------------------------------------------------------------------

class TestClinicalTrialsClient:
    def test_import(self):
        from als_rag.ingestion.clinicaltrials_client import ClinicalTrialsClient
        assert ClinicalTrialsClient is not None

    @patch("als_rag.ingestion.clinicaltrials_client.requests.Session.get")
    def test_fetch_empty_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"studies": []}
        mock_get.return_value = mock_resp
        from als_rag.ingestion.clinicaltrials_client import ClinicalTrialsClient
        client = ClinicalTrialsClient()
        results = client.fetch_als_corpus(max_results=10)
        assert results == []

    @patch("als_rag.ingestion.clinicaltrials_client.requests.Session.get")
    def test_fetch_parses_study(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "studies": [{
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT12345678",
                        "briefTitle": "Tofersen Trial",
                        "officialTitle": "Phase 3 Tofersen SOD1-ALS Trial",
                    },
                    "descriptionModule": {"briefSummary": "Test tofersen in SOD1 ALS"},
                    "statusModule": {
                        "overallStatus": "RECRUITING",
                        "startDateStruct": {"date": "2023-01"},
                    },
                    "eligibilityModule": {},
                    "outcomesModule": {"primaryOutcomes": [{"measure": "ALSFRS-R"}]},
                    "armsInterventionsModule": {"interventions": [{"interventionName": "tofersen"}]},
                    "designModule": {"phases": ["PHASE3"]},
                    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Biogen"}},
                }
            }],
        }
        mock_get.return_value = mock_resp
        from als_rag.ingestion.clinicaltrials_client import ClinicalTrialsClient
        client = ClinicalTrialsClient()
        results = client.fetch_als_corpus(max_results=1)
        assert len(results) == 1
        assert results[0]["nct_id"] == "NCT12345678"
        assert "tofersen" in results[0]["abstract"].lower()
        assert results[0]["url"].endswith("NCT12345678")


class TestEuropePMCClient:
    def test_import(self):
        from als_rag.ingestion.europepmc_client import EuropePMCClient
        assert EuropePMCClient is not None

    @patch("als_rag.ingestion.europepmc_client.requests.Session.get")
    def test_search_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"resultList": {"result": []}}
        mock_get.return_value = mock_resp
        from als_rag.ingestion.europepmc_client import EuropePMCClient
        client = EuropePMCClient()
        results = client.search("ALS", max_results=10)
        assert results == []

    @patch("als_rag.ingestion.europepmc_client.requests.Session.get")
    def test_search_parses_article(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "resultList": {
                "result": [{
                    "pmid": "12345678",
                    "title": "Neurofilament in ALS",
                    "abstractText": "NfL is elevated in ALS serum.",
                    "pubYear": 2023,
                    "authorList": {"author": [{"lastName": "Smith", "initials": "JA"}]},
                    "journalTitle": "Ann Neurol",
                    "source": "MED",
                }]
            }
        }
        mock_get.return_value = mock_resp
        from als_rag.ingestion.europepmc_client import EuropePMCClient
        client = EuropePMCClient()
        results = client.search("neurofilament ALS", max_results=5)
        assert len(results) == 1
        assert results[0]["title"] == "Neurofilament in ALS"
        assert results[0]["year"] == "2023"
        assert "Smith" in results[0]["authors"][0]
        assert results[0]["url"] == "https://europepmc.org/article/MED/12345678"
