"""
ClinicalMatchingAgent — case-based literature retrieval from clinical data.

Accepts structured clinical records (EMG, ALSFRS-R, FVC, genetics) and
returns literature most relevant to that specific patient phenotype.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClinicalMatchResult:
    """Structured output from the ClinicalMatchingAgent."""

    features_description: str
    onset_phenotype: str
    progression_rate: Optional[float]
    sources: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""

    def summary(self) -> str:
        lines = [
            f"Phenotype:  {self.onset_phenotype}",
            f"Progression: {self.progression_rate:.2f} pts/month" if self.progression_rate else "Progression: N/A",
            f"Query:      {self.features_description}",
            f"Sources:    {len(self.sources)} retrieved",
        ]
        return "\n".join(lines)


class ClinicalMatchingAgent:
    """
    Case-based literature retrieval driven by structured ALS clinical data.

    Converts EMG, ALSFRS-R, FVC, genetics, and imaging data into a
    phenotype-specific retrieval query, runs hybrid retrieval, and
    optionally generates a clinical evidence summary.

    Usage:
        agent = ClinicalMatchingAgent()

        record = {
            "alsfrs_r_total": 36,
            "fvc_percent_predicted": 68,
            "c9orf72_repeat": True,
            "denervation_regions": ["bulbar", "cervical"],
            "alsfrs_r_slope": -1.2,
        }
        result = agent.match(record)
        print(result.summary())
        print(result.answer)
    """

    def __init__(self, config=None, top_k: int = 12, generate: bool = True):
        if config is None:
            from als_rag.utils.config import get_config
            config = get_config()
        self.config = config
        self.top_k = top_k
        self.generate = generate
        self._extractor = None
        self._retriever = None
        self._generator = None
        self._ner = None

    def _get_extractor(self):
        if self._extractor is None:
            from als_rag.signals.als_matcher import ALSFeatureExtractor
            self._extractor = ALSFeatureExtractor()
        return self._extractor

    def _get_retriever(self):
        if self._retriever is None:
            from als_rag.retrieval.hybrid_retriever import HybridRetriever
            self._retriever = HybridRetriever(self.config)
        return self._retriever

    def _get_generator(self):
        if self._generator is None:
            import os
            from als_rag.generation.generator import ALSGenerator
            self._generator = ALSGenerator(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=self.config.openai_model,
            )
        return self._generator

    def _get_ner(self):
        if self._ner is None:
            from als_rag.ingestion.ner_extractor import ALSNERExtractor
            self._ner = ALSNERExtractor()
        return self._ner

    def match(self, clinical_record: Dict[str, Any]) -> ClinicalMatchResult:
        """
        Match a clinical record to relevant ALS literature.

        Args:
            clinical_record: Dict with clinical measurement keys (see
                ALSFeatureExtractor.extract_from_clinical_dict for full schema).

        Returns:
            ClinicalMatchResult with phenotype, sources, entities, and answer.
        """
        extractor = self._get_extractor()
        features = extractor.extract_from_clinical_dict(clinical_record)
        phenotype = extractor.classify_onset_phenotype(features)
        query = features.to_text_description()

        # Compute progression rate from longitudinal scores if provided
        progression_rate = None
        scores = clinical_record.get("alsfrs_r_series", [])
        times = clinical_record.get("alsfrs_r_times_months", [])
        if scores and times and len(scores) == len(times) and len(scores) >= 2:
            progression_rate = extractor.compute_progression_rate(scores, times)

        logger.info(f"ClinicalMatchingAgent: phenotype={phenotype}, query={query[:80]}")

        retriever = self._get_retriever()
        sources = retriever.retrieve(query, top_k=self.top_k)

        ner = self._get_ner()
        entities = []
        for src in sources[:3]:
            text = src.get("chunk_text", "") + " " + src.get("title", "")
            entities.extend(ner.to_dict_list(ner.extract(text)))

        # Augment query with phenotype context for generation
        enriched_query = (
            f"Clinical context: {query} (Onset phenotype: {phenotype}). "
            "What does the ALS literature say about prognosis, "
            "management, and relevant clinical trials for this presentation?"
        )

        answer = ""
        if self.generate and sources:
            generator = self._get_generator()
            answer = generator.generate(enriched_query, sources)
        elif not sources:
            answer = "No relevant literature found. Run `make ingest` first."

        return ClinicalMatchResult(
            features_description=query,
            onset_phenotype=phenotype,
            progression_rate=progression_rate,
            sources=sources,
            entities=entities,
            answer=answer,
        )

    def match_from_dict(self, record: Dict[str, Any]) -> ClinicalMatchResult:
        """Alias for match(); accepts the same clinical record dict."""
        return self.match(record)
