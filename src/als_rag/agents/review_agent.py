"""
SystematicReviewAgent — structured evidence synthesis across an ALS topic.

Queries the corpus from multiple angles, deduplicates, categorises
results by entity type, and synthesises a structured evidence summary
with sections: Background, Evidence, Gaps, and Clinical Implications.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Structured sub-questions to cover a topic from multiple angles
_REVIEW_SUB_QUERIES = [
    "{topic} epidemiology prevalence incidence",
    "{topic} pathophysiology mechanism",
    "{topic} clinical trial randomized controlled",
    "{topic} biomarker diagnosis prognosis",
    "{topic} treatment therapy outcome",
    "{topic} genetics molecular",
]

_SYNTHESIS_PROMPT = """You are an expert ALS research analyst performing a mini systematic review.

Topic: {topic}

You have been provided {n_sources} literature excerpts from {n_queries} search angles covering:
epidemiology, pathophysiology, clinical trials, biomarkers, treatments, and genetics.

Write a structured evidence summary with the following sections:

## Background
A concise 2-3 sentence overview of the topic in ALS research.

## Evidence Summary
Bullet-point synthesis of key findings from the literature, citing titles and years.
Group by theme (e.g. clinical, molecular, therapeutic).

## Evidence Gaps
What is not well established? Where is conflicting evidence? What questions remain?

## Clinical Implications
What do these findings mean for ALS patient care or clinical trial design?

Be precise. Cite source titles and years. Do not speculate beyond what the literature supports."""


@dataclass
class ReviewSection:
    """A single section of the systematic review output."""
    heading: str
    content: str


@dataclass
class SystematicReviewResult:
    """Structured output from the SystematicReviewAgent."""

    topic: str
    synthesis: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sub_queries_run: List[str] = field(default_factory=list)
    entity_counts: Dict[str, int] = field(default_factory=dict)

    def format_entity_summary(self) -> str:
        if not self.entity_counts:
            return "No entities extracted."
        lines = [f"  {label}: {count}" for label, count in sorted(self.entity_counts.items())]
        return "\n".join(lines)

    def format_source_table(self) -> str:
        lines = []
        for i, src in enumerate(self.sources, 1):
            title = src.get("title", "Unknown")[:60]
            year = src.get("year", "")
            score = src.get("score", 0.0)
            source = src.get("source", "")
            lines.append(f"{i:3}. [{score:.3f}] {title} ({year}) [{source}]")
        return "\n".join(lines)


class SystematicReviewAgent:
    """
    Automated mini systematic review for an ALS topic.

    Runs structured multi-angle retrieval, deduplicates across sub-queries,
    extracts and tallies NER entities, then synthesises a structured
    evidence summary via the ALSGenerator.

    Usage:
        agent = SystematicReviewAgent()
        result = agent.review("tofersen SOD1 ALS")
        print(result.synthesis)
        print(result.format_entity_summary())
        print(result.format_source_table())
    """

    def __init__(
        self,
        config=None,
        top_k_per_query: int = 8,
        max_total_sources: int = 20,
        generate: bool = True,
    ):
        if config is None:
            from als_rag.utils.config import get_config
            config = get_config()
        self.config = config
        self.top_k_per_query = top_k_per_query
        self.max_total_sources = max_total_sources
        self.generate = generate
        self._retriever = None
        self._generator = None
        self._ner = None

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

    def review(self, topic: str) -> SystematicReviewResult:
        """
        Run a structured mini systematic review for the given ALS topic.

        Args:
            topic: The ALS research topic (e.g. "neurofilament biomarker",
                   "tofersen SOD1", "C9orf72 frontotemporal dementia").

        Returns:
            SystematicReviewResult with synthesis text, sources, and entities.
        """
        logger.info(f"SystematicReviewAgent: topic='{topic}'")

        retriever = self._get_retriever()
        ner = self._get_ner()

        # Run multi-angle sub-queries
        all_results: Dict[str, Dict[str, Any]] = {}
        sub_queries_run: List[str] = []

        for template in _REVIEW_SUB_QUERIES:
            sub_q = template.format(topic=topic)
            sub_queries_run.append(sub_q)
            results = retriever.retrieve(sub_q, top_k=self.top_k_per_query)
            for r in results:
                key = r.get("doc_id", r.get("title", ""))[:60]
                if key not in all_results or r["score"] > all_results[key]["score"]:
                    all_results[key] = r

        # Sort by score and cap total
        ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        top_sources = ranked[:self.max_total_sources]

        # NER pass over top sources
        all_entities: List[Dict[str, Any]] = []
        entity_counts: Dict[str, int] = {}
        for src in top_sources:
            text = src.get("chunk_text", "") + " " + src.get("title", "")
            entities = ner.extract(text)
            for e in entities:
                entity_counts[e.label] = entity_counts.get(e.label, 0) + 1
            all_entities.extend(ner.to_dict_list(entities))

        # Generate structured synthesis
        synthesis = ""
        if self.generate and top_sources:
            generator = self._get_generator()
            prompt = _SYNTHESIS_PROMPT.format(
                topic=topic,
                n_sources=len(top_sources),
                n_queries=len(sub_queries_run),
            )
            synthesis = generator.generate(prompt, top_sources, max_tokens=2000)
        elif not top_sources:
            synthesis = (
                f"No literature found for topic '{topic}'. "
                "Run `make ingest` to build the corpus first."
            )
        else:
            synthesis = "(Generation disabled — retrieval-only mode)"

        return SystematicReviewResult(
            topic=topic,
            synthesis=synthesis,
            sources=top_sources,
            entities=all_entities,
            sub_queries_run=sub_queries_run,
            entity_counts=entity_counts,
        )
