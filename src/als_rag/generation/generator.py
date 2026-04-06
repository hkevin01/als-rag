"""OpenAI-based RAG answer generator for ALS research queries."""

import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

ALS_SYSTEM_PROMPT = """You are an expert ALS (Amyotrophic Lateral Sclerosis) research assistant
with deep knowledge of motor neuron disease, clinical trials, biomarkers, and genetics.

You provide evidence-based answers grounded in the scientific literature excerpts provided.
Always cite the source titles and years when referencing specific findings.
If the provided context does not contain enough information, clearly state that.
Do not speculate beyond what the literature supports.

Key domain focus areas:
- ALS genetics (SOD1, C9orf72, FUS, TARDBP, TBK1, NEK1, UBQLN2)
- Biomarkers (neurofilament light chain, TDP-43, phospho-NfH, YKL-40)
- Clinical scales (ALSFRS-R, FVC, King's staging, MiToS staging)
- Treatments (riluzole, edaravone, tofersen/BIIB067, AMX0035, gene therapies)
- Phenotypes (bulbar onset, limb onset, respiratory onset, ALS-FTD, PLS, PMA)
- Pathophysiology (TDP-43 aggregation, axonal transport, neuroinflammation, oxidative stress)
"""


def _format_context(results: List[Dict[str, Any]], max_context_chars: int = 8000) -> str:
    parts = []
    total = 0
    for r in results:
        title = r.get("title", "")
        year = r.get("year", "")
        chunk = r.get("chunk_text", "")
        snippet = f"[{title} ({year})]\n{chunk}\n"
        if total + len(snippet) > max_context_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n---\n".join(parts)


class ALSGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> str:
        if not context:
            return "No relevant literature found in the index. Please ingest papers first."

        context_text = _format_context(context)
        user_message = (
            f"Research question: {query}\n\n"
            f"Relevant literature excerpts:\n\n{context_text}\n\n"
            "Please provide a comprehensive, evidence-based answer citing the sources above."
        )

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ALS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Generation error: {e}"
