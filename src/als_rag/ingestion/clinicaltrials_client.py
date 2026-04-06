"""ClinicalTrials.gov API v2 client for ALS trial ingestion."""

import logging
import time
from typing import List, Dict, Any
import requests

logger = logging.getLogger(__name__)

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"

ALS_TRIAL_CONDITIONS = [
    "Amyotrophic Lateral Sclerosis",
    "ALS",
    "Motor Neuron Disease",
]

ALS_TRIAL_INTERVENTIONS = [
    "SOD1",
    "C9orf72",
    "tofersen",
    "AMX0035",
    "riluzole",
    "edaravone",
    "gene therapy",
    "antisense oligonucleotide",
]

ACTIVE_STATUSES = [
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "NOT_YET_RECRUITING",
    "COMPLETED",
]


class ClinicalTrialsClient:
    """
    Fetch ALS clinical trials from ClinicalTrials.gov REST API v2.

    No API key required. Free to use for research purposes.
    Rate limit: ~10 req/s (unauthenticated).
    """

    BASE_URL = CTGOV_API

    def __init__(self, rate_limit: float = 0.2):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "als-rag-research/0.1 (research use)"
        self.rate_limit = rate_limit
        self._last = 0.0

    def _wait(self):
        elapsed = time.time() - self._last
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last = time.time()

    def search_als_trials(
        self,
        max_results: int = 200,
        statuses: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search ClinicalTrials.gov for ALS trials.

        Args:
            max_results: Maximum number of trials to fetch.
            statuses: List of recruitment status filters (default: all active).

        Returns:
            List of article-format dicts suitable for the ingestion pipeline.
        """
        if statuses is None:
            statuses = ACTIVE_STATUSES

        articles: List[Dict[str, Any]] = []
        seen: set = set()
        page_token: str | None = None

        params: Dict[str, Any] = {
            "query.cond": "Amyotrophic Lateral Sclerosis",
            "filter.overallStatus": ",".join(statuses),
            "fields": (
                "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,"
                "OverallStatus,StartDate,CompletionDate,InterventionName,"
                "EligibilityCriteria,PrimaryOutcomeMeasure,Phase,StudyType,"
                "EnrollmentCount,Sponsor"
            ),
            "pageSize": min(100, max_results),
            "format": "json",
        }

        while len(articles) < max_results:
            if page_token:
                params["pageToken"] = page_token
            else:
                params.pop("pageToken", None)

            self._wait()
            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"ClinicalTrials.gov fetch failed: {e}")
                break

            studies = data.get("studies", [])
            if not studies:
                break

            for study in studies:
                pms = study.get("protocolSection", {})
                id_mod = pms.get("identificationModule", {})
                desc_mod = pms.get("descriptionModule", {})
                status_mod = pms.get("statusModule", {})
                eligibility_mod = pms.get("eligibilityModule", {})
                outcomes_mod = pms.get("outcomesModule", {})
                arms_mod = pms.get("armsInterventionsModule", {})
                design_mod = pms.get("designModule", {})
                sponsor_mod = pms.get("sponsorCollaboratorsModule", {})

                nct_id = id_mod.get("nctId", "")
                if nct_id in seen:
                    continue
                seen.add(nct_id)

                title = id_mod.get("officialTitle") or id_mod.get("briefTitle", "")
                summary = desc_mod.get("briefSummary", "")
                detailed = desc_mod.get("detailedDescription", "")
                eligibility = eligibility_mod.get("eligibilityCriteria", "")
                primary_outcomes = [
                    o.get("measure", "")
                    for o in outcomes_mod.get("primaryOutcomes", [])
                ]
                interventions = [
                    i.get("interventionName", "")
                    for i in arms_mod.get("interventions", [])
                ]
                phase = ", ".join(design_mod.get("phases", []) or [])
                sponsor = sponsor_mod.get("leadSponsor", {}).get("name", "")
                start_date = status_mod.get("startDateStruct", {}).get("date", "")
                status = status_mod.get("overallStatus", "")

                # Build a rich abstract combining all available text
                abstract_parts = [summary]
                if detailed:
                    abstract_parts.append(f"Details: {detailed[:500]}")
                if eligibility:
                    abstract_parts.append(f"Eligibility: {eligibility[:300]}")
                if primary_outcomes:
                    abstract_parts.append(f"Primary outcomes: {'; '.join(primary_outcomes[:3])}")
                if interventions:
                    abstract_parts.append(f"Interventions: {', '.join(interventions[:5])}")
                if phase:
                    abstract_parts.append(f"Phase: {phase}")

                articles.append({
                    "title": title,
                    "abstract": " ".join(abstract_parts).strip(),
                    "year": start_date[:4] if start_date else "",
                    "authors": [sponsor] if sponsor else [],
                    "journal": f"ClinicalTrials.gov {status}",
                    "url": f"https://clinicaltrials.gov/study/{nct_id}",
                    "source": "clinicaltrials",
                    "nct_id": nct_id,
                    "phase": phase,
                    "status": status,
                })

                if len(articles) >= max_results:
                    break

            next_token = data.get("nextPageToken")
            if not next_token:
                break
            page_token = next_token

        logger.info(f"ClinicalTrials.gov: fetched {len(articles)} ALS trials")
        return articles

    def fetch_als_corpus(self, max_results: int = 200) -> List[Dict[str, Any]]:
        """Fetch ALS trials corpus for indexing."""
        return self.search_als_trials(max_results=max_results)
