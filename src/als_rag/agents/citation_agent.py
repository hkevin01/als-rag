"""
CitationVerificationAgent — hallucination guard for ALS-RAG answers.

Checks each sentence in a generated answer against the source passages
that were retrieved, scoring how well each claim is backed by evidence.
Flags unsupported sentences that may represent LLM hallucinations.

Strategy:
    1. Tokenise the answer into individual claim sentences.
    2. For every claim, compute domain-weighted lexical overlap against
       every source chunk_text + title (Jaccard on content words).
    3. A claim is "supported" if its best source overlap >= threshold.
    4. Return a CitationVerificationResult with per-claim verdicts,
       an overall coverage score, and a flagged list of weak citations.

This is an intentionally lightweight, offline check — no extra API calls.
For a model-based NLI approach, swap _compute_overlap() with an NLI scorer.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Common English stop words — excluded from overlap computation so that
# domain-specific terms (genes, biomarkers, drugs) carry more weight.
_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of with by from is are was were be been"
    " being have has had do does did will would could should may might shall not"
    " no nor so yet both either neither it its this that these those he she they"
    " we you i me him her us them what which who whom whose when where why how"
    " all any each few more most other some such than then there their can also"
    " into through during before after above below between out off over under"
    " again further then once here there when if because as until while".split()
)


def _tokenise(text: str) -> frozenset:
    """Lower-case, strip punctuation, remove stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(w for w in words if w not in _STOP_WORDS and len(w) > 1)


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard index: |intersection| / |union|."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _coverage_overlap(claim_tokens: frozenset, source_tokens: frozenset) -> float:
    """Claim-coverage: fraction of claim tokens found in the source."""
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & source_tokens) / len(claim_tokens)


def _split_into_sentences(text: str) -> List[str]:
    """
    Split on sentence boundaries.  Handles:
      - "Title (Year)." patterns (citations)
      - Bullet lines starting with "-" or "*"
      - Numbered list items "1. ..."
    Returns only sentences >= 6 words.
    """
    # Replace newlines with spaces for uniform processing
    flat = text.replace("\n", " ")
    # Split on ". " or "! " or "? " followed by capital or digit
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\d\"])", flat)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s.split()) >= 4:
            sentences.append(s)
    return sentences if sentences else [text.strip()]


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class CitationClaim:
    """Represents a single sentence / claim extracted from the answer."""

    text: str
    supported: bool
    best_source_title: Optional[str]
    best_score: float   # 0–1 claim-coverage overlap score

    def verdict(self) -> str:
        if self.supported:
            return f"✅ Supported  [{self.best_score:.2f}] — {self.best_source_title or 'unknown'}"
        return f"⚠️  Unsupported [{self.best_score:.2f}]"


@dataclass
class CitationVerificationResult:
    """Structured output from the CitationVerificationAgent."""

    answer: str
    claims: List[CitationClaim] = field(default_factory=list)
    coverage_score: float = 0.0      # 0–1 fraction of supported claims
    unsupported: List[str] = field(default_factory=list)   # texts of flagged sentences
    flagged: bool = False             # True if coverage < threshold (default 0.5)
    threshold: float = 0.5

    def report(self) -> str:
        """Return a human-readable verification report."""
        lines = [
            f"Citation Coverage: {self.coverage_score:.0%} "
            f"({'PASS' if not self.flagged else 'FLAG — review unsupported claims'})",
            f"Claims checked: {len(self.claims)}  |  "
            f"Supported: {sum(1 for c in self.claims if c.supported)}  |  "
            f"Unsupported: {len(self.unsupported)}",
            "",
        ]
        for i, claim in enumerate(self.claims, 1):
            lines.append(f"  [{i:2}] {claim.verdict()}")
            lines.append(f"       › {claim.text[:100]}{'...' if len(claim.text) > 100 else ''}")
        if self.unsupported:
            lines += ["", "━ Unsupported claims (potential hallucinations) ━"]
            for u in self.unsupported:
                lines.append(f"  ⚠  {u[:120]}{'...' if len(u) > 120 else ''}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

class CitationVerificationAgent:
    """
    Offline citation / hallucination checker for ALS-RAG answers.

    Checks each sentence of a generated answer against the retrieved
    source passages using domain-weighted lexical overlap.  No extra
    API calls required.

    Why this exists:
        LLMs can confidently state incorrect or unsourced facts
        (hallucinations).  ALS-RAG is grounded via RAG, but the
        generator may still synthesize details not present in the
        retrieved context.  This agent provides a fast, interpretable
        audit trail showing exactly which sentences are backed by which
        papers and which are not.

    Usage:
        from als_rag.agents import ResearchAgent, CitationVerificationAgent

        research = ResearchAgent()
        result = research.ask("What is the prognostic role of NfL in ALS?")

        verifier = CitationVerificationAgent()
        vresult = verifier.verify(result.answer, result.sources)
        print(vresult.report())
    """

    def __init__(
        self,
        support_threshold: float = 0.12,
        coverage_flag_threshold: float = 0.50,
    ):
        """
        Args:
            support_threshold: Minimum claim-coverage overlap for a sentence
                to be considered "supported" by a source (default 0.12).
                Increase for stricter checking; decrease for shorter claims.
            coverage_flag_threshold: If fraction of supported claims falls
                below this, the overall result is flagged (default 0.50).
        """
        self.support_threshold = support_threshold
        self.coverage_flag_threshold = coverage_flag_threshold

    def verify(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> CitationVerificationResult:
        """
        Verify that each sentence in the answer is backed by a source.

        Args:
            answer: The generated answer text to verify.
            sources: List of retrieved source dicts (must have 'chunk_text'
                     and/or 'title' keys).  Typically result.sources from
                     ResearchAgent or SystematicReviewAgent.

        Returns:
            CitationVerificationResult with per-claim verdicts.
        """
        if not answer or not answer.strip():
            logger.warning("CitationVerificationAgent: empty answer provided")
            return CitationVerificationResult(answer=answer, coverage_score=0.0, flagged=True)

        # Pre-tokenise all sources once
        source_token_sets = []
        for src in sources:
            text = (src.get("chunk_text", "") + " " + src.get("title", "")).strip()
            source_token_sets.append((src.get("title", "Unknown"), _tokenise(text)))

        sentences = _split_into_sentences(answer)
        logger.info(
            f"CitationVerificationAgent: checking {len(sentences)} claims "
            f"against {len(sources)} sources"
        )

        claims: List[CitationClaim] = []
        for sentence in sentences:
            claim_tokens = _tokenise(sentence)
            best_score = 0.0
            best_title = None

            for title, src_tokens in source_token_sets:
                score = _coverage_overlap(claim_tokens, src_tokens)
                if score > best_score:
                    best_score = score
                    best_title = title

            supported = best_score >= self.support_threshold
            claims.append(CitationClaim(
                text=sentence,
                supported=supported,
                best_source_title=best_title,
                best_score=best_score,
            ))

        n_supported = sum(1 for c in claims if c.supported)
        coverage = n_supported / len(claims) if claims else 0.0
        unsupported = [c.text for c in claims if not c.supported]
        flagged = coverage < self.coverage_flag_threshold

        result = CitationVerificationResult(
            answer=answer,
            claims=claims,
            coverage_score=coverage,
            unsupported=unsupported,
            flagged=flagged,
            threshold=self.coverage_flag_threshold,
        )

        if flagged:
            logger.warning(
                f"CitationVerificationAgent: coverage {coverage:.0%} below threshold "
                f"({self.coverage_flag_threshold:.0%}) — {len(unsupported)} unsupported claims"
            )
        else:
            logger.info(f"CitationVerificationAgent: coverage {coverage:.0%} — PASS")

        return result

    def verify_from_research_result(self, result) -> CitationVerificationResult:
        """
        Convenience wrapper: accept a ResearchResult or SystematicReviewResult
        directly (any object with .answer and .sources attributes).
        """
        return self.verify(result.answer, result.sources)
