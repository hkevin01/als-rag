"""Tests for ALSFeatureExtractor."""

import pytest
from als_rag.signals.als_matcher import ALSFeatureExtractor, ALSFeatures


def _default_features(**kwargs) -> ALSFeatures:
    defaults = dict(
        fasciculation_score=1.0,
        denervation_regions=["cervical"],
        motor_unit_amplitude=500.0,
        recruitment_pattern="reduced",
        alsfrs_r_score=40,
        alsfrs_r_slope=-0.5,
        disease_duration_months=12.0,
        fvc_percent=90.0,
        sniff_nasal_pressure=60.0,
    )
    defaults.update(kwargs)
    return ALSFeatures(**defaults)


@pytest.fixture
def extractor():
    return ALSFeatureExtractor()


def test_classify_bulbar_onset(extractor):
    features = _default_features(denervation_regions=["bulbar", "cervical"])
    phenotype = extractor.classify_onset_phenotype(features)
    assert isinstance(phenotype, str)
    assert len(phenotype) > 0


def test_classify_limb_onset(extractor):
    features = _default_features(denervation_regions=["lumbar"])
    phenotype = extractor.classify_onset_phenotype(features)
    assert isinstance(phenotype, str)


def test_extract_from_clinical_dict(extractor):
    data = {
        "alsfrs_r_total": 38,
        "fvc_percent_predicted": 85,
        "nfl_pg_ml": 120.0,
        "c9orf72_repeat": False,
        "sod1_variant": False,
    }
    features = extractor.extract_from_clinical_dict(data)
    assert isinstance(features, ALSFeatures)
    assert features.alsfrs_r_score == 38
    assert features.fvc_percent == 85.0


def test_progression_rate(extractor):
    scores = [48, 44, 40, 36, 32]
    timestamps = [0, 1, 2, 3, 4]
    rate = extractor.compute_progression_rate(scores, timestamps)
    assert isinstance(rate, float)
    assert rate < 0  # declining


def test_to_text_description():
    features = _default_features(alsfrs_r_score=40, fvc_percent=90)
    text = features.to_text_description()
    assert isinstance(text, str)
    assert len(text) > 0
