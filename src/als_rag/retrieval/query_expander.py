"""ALS-specific query expansion: adds synonyms and related terms."""

from typing import List

ALS_SYNONYMS: dict = {
    "als": ["amyotrophic lateral sclerosis", "lou gehrig disease", "motor neuron disease"],
    "amyotrophic lateral sclerosis": ["ALS", "MND", "Lou Gehrig"],
    "tdp-43": ["TARDBP", "TDP43", "TAR DNA binding protein 43"],
    "neurofilament": ["NfL", "neurofilament light chain", "NfH", "pNfH"],
    "sod1": ["SOD1 mutation", "copper zinc superoxide dismutase"],
    "c9orf72": ["C9orf72 repeat expansion", "chromosome 9 open reading frame 72"],
    "alsfrs": ["ALSFRS-R", "ALS functional rating scale revised"],
    "riluzole": ["Rilutek", "glutamate antagonist ALS"],
    "tofersen": ["BIIB067", "antisense oligonucleotide SOD1"],
    "biomarker": ["neurofilament", "NfL", "TDP-43", "phospho-neurofilament"],
    "progression": ["disease progression", "ALSFRS-R decline", "survival ALS"],
    "gene therapy": ["AAV therapy ALS", "antisense oligonucleotide ALS"],
}


class ALSQueryExpander:
    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        expanded = [query]
        query_lower = query.lower()
        count = 0
        for term, synonyms in ALS_SYNONYMS.items():
            if term.lower() in query_lower and count < max_expansions:
                for syn in synonyms[:2]:
                    if syn.lower() not in query_lower:
                        expanded.append(query.lower().replace(term, syn))
                        count += 1
        return list(dict.fromkeys(expanded))  # deduplicate, preserve order
