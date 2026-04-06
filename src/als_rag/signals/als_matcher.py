"""
ALS Clinical Signal Integration.

Provides case-based retrieval by matching ALS biomarker and clinical features
with literature. Analogous to eeg_matcher.py but for ALS-specific signals:
EMG, nerve conduction, neuroimaging markers, and clinical progression metrics.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ALSFeatures:
    """Extracted ALS biomarker and clinical features for literature matching."""

    # EMG / Nerve Conduction
    fasciculation_score: float          # 0-4 Awaji criteria score
    denervation_regions: List[str]      # e.g. ["bulbar", "cervical", "lumbar"]
    motor_unit_amplitude: float         # µV — mean MUAP amplitude
    recruitment_pattern: str            # "reduced", "normal", "increased"

    # Clinical progression metrics
    alsfrs_r_score: int                 # ALS Functional Rating Scale-Revised (0–48)
    alsfrs_r_slope: float              # Points/month decline
    disease_duration_months: float

    # Respiratory
    fvc_percent: float                  # Forced Vital Capacity % predicted
    sniff_nasal_pressure: float         # cmH2O

    # Biomarkers
    neurofilament_light: Optional[float] = None  # NfL pg/mL (CSF or serum)
    tdp43_pathology: bool = False
    sod1_variant: bool = False
    c9orf72_repeat: bool = False
    fus_variant: bool = False

    # Cognitive / Behaviour
    cognitive_impairment: bool = False
    behavioural_variant: bool = False   # ALS-FTD overlap

    # Imaging
    upper_motor_neuron_signs: bool = True
    lower_motor_neuron_signs: bool = True
    corticospinal_involvement: Dict[str, float] = field(default_factory=dict)

    def to_text_description(self) -> str:
        """Convert features to natural language query for literature retrieval."""
        parts = []

        # Onset region
        if self.denervation_regions:
            regions = ", ".join(self.denervation_regions)
            parts.append(f"denervation in {regions} regions")

        # Genetics
        if self.c9orf72_repeat:
            parts.append("C9orf72 repeat expansion")
        if self.sod1_variant:
            parts.append("SOD1 mutation")
        if self.fus_variant:
            parts.append("FUS variant")
        if self.tdp43_pathology:
            parts.append("TDP-43 pathology")

        # Progression
        if self.alsfrs_r_slope > 1.0:
            parts.append("rapid progression (ALSFRS-R slope > 1 pt/month)")
        elif self.alsfrs_r_slope < 0.3:
            parts.append("slow progression")

        # Respiratory
        if self.fvc_percent < 50:
            parts.append("severe respiratory compromise (FVC < 50%)")
        elif self.fvc_percent < 70:
            parts.append("moderate respiratory compromise")

        # NfL
        if self.neurofilament_light and self.neurofilament_light > 100:
            parts.append("elevated serum neurofilament light chain")

        # Cognitive overlap
        if self.cognitive_impairment:
            parts.append("cognitive impairment / ALS-FTD overlap")

        # UMN/LMN
        signs = []
        if self.upper_motor_neuron_signs:
            signs.append("upper motor neuron")
        if self.lower_motor_neuron_signs:
            signs.append("lower motor neuron")
        if signs:
            parts.append(f"{' and '.join(signs)} signs present")

        base = "ALS case with " + (", ".join(parts) if parts else "typical presentation")
        return base


class ALSFeatureExtractor:
    """
    Extract ALS-relevant features from structured clinical data inputs.

    Accepts structured dicts (from CSV, EHR, REDCap export, etc.) and returns
    an ALSFeatures object suitable for literature retrieval queries.
    """

    # ALSFRS-R subscore domains
    ALSFRS_DOMAINS = {
        "bulbar": ["speech", "salivation", "swallowing"],
        "fine_motor": ["handwriting", "cutting_food", "dressing"],
        "gross_motor": ["turning_in_bed", "walking", "climbing_stairs"],
        "respiratory": ["dyspnea", "orthopnea", "respiratory_insufficiency"],
    }

    # Awaji criteria EMG classification
    AWAJI_CRITERIA = {
        0: "no abnormality",
        1: "chronic denervation only",
        2: "active and chronic denervation",
        3: "fasciculation potentials present",
        4: "fibrillation + positive sharp waves",
    }

    def extract_from_clinical_dict(self, record: Dict[str, Any]) -> ALSFeatures:
        """
        Extract ALSFeatures from a clinical record dictionary.

        Args:
            record: Dict with keys matching clinical data fields

        Returns:
            ALSFeatures populated from the record
        """
        logger.info("Extracting ALS features from clinical record")

        features = ALSFeatures(
            fasciculation_score=float(record.get("fasciculation_score", 0)),
            denervation_regions=record.get("denervation_regions", []),
            motor_unit_amplitude=float(record.get("muap_amplitude_uv", 500)),
            recruitment_pattern=record.get("recruitment_pattern", "reduced"),
            alsfrs_r_score=int(record.get("alsfrs_r_total", 40)),
            alsfrs_r_slope=float(record.get("alsfrs_r_slope", 0.5)),
            disease_duration_months=float(record.get("disease_duration_months", 12)),
            fvc_percent=float(record.get("fvc_percent_predicted", 90)),
            sniff_nasal_pressure=float(record.get("snp_cmh2o", 70)),
            neurofilament_light=record.get("nfl_pg_ml"),
            tdp43_pathology=bool(record.get("tdp43_pathology", False)),
            sod1_variant=bool(record.get("sod1_variant", False)),
            c9orf72_repeat=bool(record.get("c9orf72_repeat", False)),
            fus_variant=bool(record.get("fus_variant", False)),
            cognitive_impairment=bool(record.get("cognitive_impairment", False)),
            behavioural_variant=bool(record.get("behavioural_variant", False)),
            upper_motor_neuron_signs=bool(record.get("umn_signs", True)),
            lower_motor_neuron_signs=bool(record.get("lmn_signs", True)),
            corticospinal_involvement=record.get("corticospinal_involvement", {}),
        )

        logger.info(
            f"Features extracted: ALSFRS-R={features.alsfrs_r_score}, "
            f"FVC={features.fvc_percent:.0f}%, regions={features.denervation_regions}"
        )
        return features

    def compute_progression_rate(
        self, scores: List[int], timestamps_months: List[float]
    ) -> float:
        """
        Compute ALSFRS-R slope (points/month) from longitudinal scores.

        Args:
            scores: List of ALSFRS-R total scores
            timestamps_months: Corresponding times in months from onset

        Returns:
            Slope in points/month (negative = decline)
        """
        if len(scores) < 2:
            return 0.0
        x = np.array(timestamps_months)
        y = np.array(scores)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def classify_onset_phenotype(self, features: ALSFeatures) -> str:
        """
        Classify ALS onset phenotype for targeted literature search.

        Returns one of: 'bulbar', 'limb_onset', 'respiratory_onset',
                        'flail_arm', 'flail_leg', 'als_ftd'
        """
        if features.cognitive_impairment or features.behavioural_variant:
            return "als_ftd"
        if "bulbar" in features.denervation_regions and (
            "cervical" not in features.denervation_regions
            and "lumbar" not in features.denervation_regions
        ):
            return "bulbar"
        if features.fvc_percent < 65 and features.alsfrs_r_score > 35:
            return "respiratory_onset"
        if "cervical" in features.denervation_regions:
            return "flail_arm"
        if "lumbar" in features.denervation_regions:
            return "flail_leg"
        return "limb_onset"
