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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    if args.ingest:
        _run_ingestion()
        if not args.question:
            return

    if not args.question:
        parser.print_help()
        sys.exit(1)

    from als_rag.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.retrieve(args.question, top_k=args.top_k)

    if not results:
        print("No results found. Run with --ingest first.")
        sys.exit(1)

    if not args.no_generate:
        from als_rag.generation.generator import ALSGenerator
        gen = ALSGenerator(api_key=os.environ.get("OPENAI_API_KEY", ""))
        answer = gen.generate(args.question, results)
        print("\n=== Answer ===")
        print(answer)
        print()

    print("=== Sources ===")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.get('title', '')} ({r.get('year', '')}) score={r.get('score', 0):.3f}")
        if r.get("url"):
            print(f"   {r['url']}")


def _run_ingestion():
    from als_rag.ingestion.pubmed_client import PubMedClient
    from als_rag.ingestion.scholar_client import ScholarClient
    from als_rag.ingestion.arxiv_client import ArxivClient
    from als_rag.ingestion.pipeline import ALSIngestionPipeline

    print("Starting ingestion...")
    pipeline = ALSIngestionPipeline()
    all_articles = []

    print("Fetching from PubMed...")
    pm = PubMedClient(
        api_key=os.environ.get("PUBMED_API_KEY"),
        contact_email=os.environ.get("CONTACT_EMAIL", "user@example.com"),
    )
    all_articles.extend(pm.fetch_als_corpus(max_per_query=30))

    print("Fetching from Semantic Scholar...")
    sc = ScholarClient(api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    all_articles.extend(sc.fetch_als_corpus(papers_per_query=30))

    print("Fetching from arXiv...")
    from als_rag.ingestion.arxiv_client import ArxivClient as Ax
    ax = Ax()
    all_articles.extend(ax.fetch_als_corpus(results_per_query=20))

    print(f"Total articles: {len(all_articles)}")
    n_chunks = pipeline.ingest(all_articles)
    print(f"Indexed {n_chunks} chunks.")


if __name__ == "__main__":
    main()
