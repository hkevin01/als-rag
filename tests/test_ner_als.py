"""Tests for ALSNERExtractor."""

import pytest
from als_rag.ingestion.ner_extractor import ALSNERExtractor


@pytest.fixture
def extractor():
    return ALSNERExtractor()


def test_extract_gene(extractor):
    text = "Mutations in SOD1 are associated with familial ALS."
    entities = extractor.extract(text)
    genes = [e for e in entities if e.label == "GENE"]
    names = [e.text.lower() for e in genes]
    assert any("sod1" in n for n in names)


def test_extract_biomarker(extractor):
    text = "Neurofilament light chain levels were elevated in ALS patients."
    entities = extractor.extract(text)
    bios = [e for e in entities if e.label == "BIOMARKER"]
    assert len(bios) >= 1


def test_extract_scale(extractor):
    text = "The ALSFRS-R score declined significantly over 6 months."
    entities = extractor.extract(text)
    scales = [e for e in entities if e.label == "CLINICAL_SCALE"]
    assert len(scales) >= 1


def test_extract_drug(extractor):
    text = "Riluzole remains the standard of care for ALS."
    entities = extractor.extract(text)
    drugs = [e for e in entities if e.label == "TREATMENT"]
    assert len(drugs) >= 1


def test_extract_subtype(extractor):
    text = "Bulbar onset ALS has a worse prognosis than limb onset."
    entities = extractor.extract(text)
    subtypes = [e for e in entities if e.label == "ALS_SUBTYPE"]
    assert len(subtypes) >= 1


def test_empty_text(extractor):
    entities = extractor.extract("")
    assert entities == []
