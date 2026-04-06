"""
ResearchAgent — end-to-end ALS literature Q&A agent.

Orchestrates the full RAG pipeline: query expansion → hybrid retrieval
→ NER annotation → evidence-grounded generation with ranked citations.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Structured output from the ResearchAgent."""

    query: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    expanded_queries: List[str] = field(default_factory=list)

    def format_citation_list(self) -> str:
        """Return a numbered citation list suitable for display."""
        lines = []
        for i, src in enumerate(self.sources, 1):
            title = src.get("title", "Unknown")
            year = src.get("year", "")
            url = src.get("url", "")
            score = src.get("score", 0.0)
            line = f"{i}. {title} ({year}) [score={score:.3f}]"
            if url:
                line += f"\n   {url}"
            lines.append(line)
        return "\n".join(lines)


class ResearchAgent:
    """
    End-to-end ALS research Q&A agent.

    Wraps HybridRetriever + ALSNERExtractor + ALSGenerator into a single
    callable. Enriches results with entity annotations and returns a
    structured ResearchResult.

    Usage:
        agent = ResearchAgent()
        result = agent.ask("What is the prognostic value of NfL in ALS?")
        print(result.answer)
        print(result.format_citation_list())
    """

    def __init__(
        self,
        config=None,
        top_k: int = 10,
        generate: bool = True,
        api_key: Optional[str] = None,
    ):
        if config is None:
            from als_rag.utils.config import get_config
            config = get_config()
        self.config = config
        self.top_k = top_k
        self.generate = generate
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._retriever = None
        self._generator = None
        self._ner = None
        self._expander = None

    def _get_retriever(self):
        if self._retriever is None:
            from als_rag.retrieval.hybrid_retriever import HybridRetriever
            self._retriever = HybridRetriever(self.config)
        return self._retriever

    def _get_generator(self):
        if self._generator is None:
            from als_rag.generation.generator import ALSGenerator
            self._generator = ALSGenerator(
                api_key=self.api_key, model=self.config.openai_model
            )
        return self._generator

    def _get_ner(self):
        if self._ner is None:
            from als_rag.ingestion.ner_extractor import ALSNERExtractor
            self._ner = ALSNERExtractor()
        return self._ner

    def _get_expander(self):
        if self._expander is None:
            from als_rag.retrieval.query_expander import ALSQueryExpander
            self._expander = ALSQueryExpander()
        return self._expander

    def ask(self, query: str) -> ResearchResult:
        """
        Run the full RAG pipeline for a research question.

        Args:
            query: Natural-language research question.

        Returns:
            ResearchResult with answer, ranked sources, and NER annotations.
        """
        logger.info(f"ResearchAgent: '{query[:80]}'")

        expander = self._get_expander()
        expanded = expander.expand(query, max_expansions=2)

        retriever = self._get_retriever()
        sources = retriever.retrieve(query, top_k=self.top_k)

        ner = self._get_ner()
        entities = []
        for src in sources[:3]:
            text = src.get("chunk_text", "") + " " + src.get("title", "")
            entities.extend(ner.to_dict_list(ner.extract(text)))

        answer = ""
        if self.generate and sources:
            generator = self._get_generator()
            answer = generator.generate(query, sources)
        elif not sources:
            answer = (
                "No relevant literature found in the index. "
                "Run `make ingest` to build the corpus first."
            )
        else:
            answer = "(Generation disabled — retrieval-only mode)"

        return ResearchResult(
            query=query,
            answer=answer,
            sources=sources,
            entities=entities,
            expanded_queries=expanded,
        )
