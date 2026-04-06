"""Tests for ALSQueryExpander."""

from als_rag.retrieval.query_expander import ALSQueryExpander


def test_expand_als():
    expander = ALSQueryExpander()
    queries = expander.expand("ALS clinical trial")
    assert len(queries) >= 1
    assert "ALS clinical trial" in queries


def test_expand_neurofilament():
    expander = ALSQueryExpander()
    queries = expander.expand("neurofilament biomarker study")
    assert len(queries) > 1  # should have expansions


def test_no_duplicate_original():
    expander = ALSQueryExpander()
    queries = expander.expand("SOD1 ALS therapy")
    assert queries.count(queries[0]) == 1


def test_expand_unknown_term():
    expander = ALSQueryExpander()
    queries = expander.expand("randomXYZ999 query")
    assert queries == ["randomXYZ999 query"]
