"""
ALS Named Entity Recognition Extractor.

Extracts ALS-specific entities from scientific text: genes, biomarkers,
clinical trials, drugs, disease subtypes, and outcome measures.
"""

import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ALSEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


# ALS-specific entity vocabularies
ALS_GENES = {
    "SOD1", "C9orf72", "C9ORF72", "FUS", "TARDBP", "TDP-43", "TDP43",
    "UBQLN2", "VCP", "OPTN", "TBK1", "SQSTM1", "HNRNPA1", "HNRNPA2B1",
    "MATR3", "TUBA4A", "NEK1", "KIF5A", "SETX", "ALS2", "ALSIN",
    "DCTN1", "CHMP2B", "ANG", "VEGF", "NEFH", "PRPH",
}

ALS_BIOMARKERS = {
    "neurofilament light", "NfL", "NFL", "pNfH", "phosphorylated neurofilament heavy",
    "TDP-43", "phospho-TDP-43", "FUS protein", "SOD1 protein",
    "chitotriosidase", "YKL-40", "CHI3L1", "CHIT1",
    "creatinine", "uric acid", "creatine kinase", "CK",
    "miR-206", "miR-133", "miR-9",
    "cytokines", "IL-6", "TNF-alpha", "MCP-1",
}

ALS_CLINICAL_SCALES = {
    "ALSFRS-R", "ALSFRS", "ALS Functional Rating Scale",
    "FVC", "forced vital capacity", "SVC", "slow vital capacity",
    "ATLIS", "SNP", "sniff nasal pressure",
    "MRC scale", "grip strength",
    "El Escorial", "Awaji criteria", "Gold Coast criteria",
}

ALS_DRUGS_TREATMENTS = {
    "riluzole", "edaravone", "tofersen", "AMX0035",
    "sodium phenylbutyrate", "tauroursodeoxycholic acid", "TUDCA",
    "rasagiline", "mexiletine", "baclofen",
    "non-invasive ventilation", "NIV", "BiPAP", "invasive ventilation",
    "PEG", "percutaneous endoscopic gastrostomy",
    "antisense oligonucleotide", "ASO", "gene therapy",
    "stem cell", "iPSC",
}

ALS_SUBTYPES = {
    "bulbar onset ALS", "limb onset ALS", "flail arm", "flail leg",
    "primary lateral sclerosis", "PLS", "progressive muscular atrophy", "PMA",
    "ALS-FTD", "ALS-frontotemporal dementia", "familial ALS", "fALS",
    "sporadic ALS", "sALS", "juvenile ALS",
    "respiratory onset ALS",
}


class ALSNERExtractor:
    """Rule-based NER for ALS research literature."""

    def extract(self, text: str) -> List[ALSEntity]:
        """Extract ALS entities from text."""
        entities: List[ALSEntity] = []
        entities.extend(self._match_vocabulary(text, ALS_GENES, "GENE"))
        entities.extend(self._match_vocabulary(text, ALS_BIOMARKERS, "BIOMARKER"))
        entities.extend(self._match_vocabulary(text, ALS_CLINICAL_SCALES, "CLINICAL_SCALE"))
        entities.extend(self._match_vocabulary(text, ALS_DRUGS_TREATMENTS, "TREATMENT"))
        entities.extend(self._match_vocabulary(text, ALS_SUBTYPES, "ALS_SUBTYPE"))
        entities.extend(self._extract_measurements(text))
        entities.extend(self._extract_sample_sizes(text))
        # Deduplicate by span
        return self._deduplicate(entities)

    def _match_vocabulary(
        self, text: str, vocab: set, label: str
    ) -> List[ALSEntity]:
        entities = []
        for term in vocab:
            pattern = r"\b" + re.escape(term) + r"\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ALSEntity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                ))
        return entities

    def _extract_measurements(self, text: str) -> List[ALSEntity]:
        """Extract numeric clinical measurements."""
        entities = []
        patterns = [
            (r"\bALSFRS-R\s+(?:score\s+)?(?:of\s+)?\d+", "ALSFRS_SCORE"),
            (r"\bFVC\s+(?:of\s+)?\d+(?:\.\d+)?\s*%", "FVC_MEASUREMENT"),
            (r"\bNfL\s+(?:of\s+)?\d+(?:\.\d+)?\s*pg/mL", "NFL_LEVEL"),
            (r"\bsurvival\s+(?:of\s+)?\d+(?:\.\d+)?\s+months", "SURVIVAL"),
        ]
        for pattern, label in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ALSEntity(
                    text=match.group(), label=label,
                    start=match.start(), end=match.end()
                ))
        return entities

    def _extract_sample_sizes(self, text: str) -> List[ALSEntity]:
        """Extract trial/study sample sizes."""
        entities = []
        for match in re.finditer(
            r"\b(?:n\s*=\s*|N\s*=\s*)(\d+)\b", text
        ):
            entities.append(ALSEntity(
                text=match.group(), label="SAMPLE_SIZE",
                start=match.start(), end=match.end()
            ))
        return entities

    def _deduplicate(self, entities: List[ALSEntity]) -> List[ALSEntity]:
        seen: set = set()
        result = []
        for e in sorted(entities, key=lambda x: x.start):
            key = (e.start, e.end)
            if key not in seen:
                seen.add(key)
                result.append(e)
        return result

    def to_dict_list(self, entities: List[ALSEntity]) -> List[Dict[str, Any]]:
        return [
            {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
            for e in entities
        ]
