"""CLI entry point: ask ALS-RAG a question."""

import argparse
import logging
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="ALS-RAG command line query")
    parser.add_argument("question", nargs="?", help="Research question")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--ingest", action="store_true", help="Ingest corpus first")
    parser.add_argument(
        "--sources",
        default="pubmed,scholar,arxiv,clinicaltrials,europepmc",
        help="Comma-separated ingestion sources (default: all five)",
    )
    parser.add_argument(
        "--review",
        metavar="TOPIC",
        help="Run a systematic mini-review on a topic (e.g. 'tofersen SOD1 ALS')",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run CitationVerificationAgent on the generated answer to flag hallucinations",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    if args.ingest:
        _run_ingestion(args.sources)
        if not args.question and not args.review:
            return

    if args.review:
        _run_review(args.review)
        return

    if not args.question:
        parser.print_help()
        sys.exit(1)

    from als_rag.agents.research_agent import ResearchAgent
    agent = ResearchAgent(
        top_k=args.top_k,
        generate=not args.no_generate,
    )
    result = agent.ask(args.question)

    if result.answer:
        print("\n=== Answer ===")
        print(result.answer)
        print()

    print("=== Sources ===")
    print(result.format_citation_list())

    if result.entities:
        labels = {}
        for e in result.entities:
            labels[e["label"]] = labels.get(e["label"], 0) + 1
        print("\n=== Entities Detected ===")
        for label, count in sorted(labels.items()):
            print(f"  {label}: {count}")

    if result.expanded_queries and len(result.expanded_queries) > 1:
        print("\n=== Query Expansions Used ===")
        for q in result.expanded_queries:
            print(f"  - {q}")

    if args.verify and result.answer:
        from als_rag.agents.citation_agent import CitationVerificationAgent
        verifier = CitationVerificationAgent()
        vresult = verifier.verify(result.answer, result.sources)
        print("\n=== Citation Verification ===")
        print(vresult.report())

def _run_ingestion(sources_str: str = "pubmed,scholar,arxiv,clinicaltrials,europepmc"):
    from als_rag.agents.ingestion_agent import IngestionAgent
    sources = [s.strip() for s in sources_str.split(",") if s.strip()]
    print(f"Starting ingestion from: {', '.join(sources)}")
    agent = IngestionAgent(sources=sources)
    report = agent.run(on_progress=print)
    print(report.summary())


def _run_review(topic: str):
    from als_rag.agents.review_agent import SystematicReviewAgent
    print(f"Running systematic review: '{topic}'")
    agent = SystematicReviewAgent()
    result = agent.review(topic)
    print("\n=== Synthesis ===")
    print(result.synthesis)
    print("\n=== Entity Distribution ===")
    print(result.format_entity_summary())
    print(f"\n=== Sources ({len(result.sources)}) ===")
    print(result.format_source_table())


if __name__ == "__main__":
    main()
