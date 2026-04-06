"""Configuration management for ALS-RAG system."""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[4]
DATA_DIR = BASE_DIR / "data"


@dataclass
class Config:
    """Central configuration for ALS-RAG."""

    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_max_tokens: int = 1500
    openai_temperature: float = 0.2

    # Data paths
    data_raw_dir: Path = DATA_DIR / "raw"
    data_processed_dir: Path = DATA_DIR / "processed"
    data_embeddings_dir: Path = DATA_DIR / "embeddings"

    # FAISS index
    faiss_index_path: Path = DATA_DIR / "embeddings" / "als_faiss.index"
    faiss_metadata_path: Path = DATA_DIR / "embeddings" / "als_metadata.json"

    # Embedding
    # Options:
    #   "sentence-transformers/all-MiniLM-L6-v2"   — fast, general (384d)
    #   "neuml/pubmedbert-base-embeddings"           — domain-tuned, biomedical (768d)
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384"))
    )

    # Retrieval
    default_top_k: int = 10
    min_similarity_score: float = 0.3

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # API rate limits
    pubmed_api_key: str = field(default_factory=lambda: os.getenv("PUBMED_API_KEY", ""))
    semantic_scholar_api_key: str = field(
        default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    )
    contact_email: str = field(
        default_factory=lambda: os.getenv("CONTACT_EMAIL", "als-rag@research.org")
    )

    # Processing
    max_workers: int = 4
    batch_size: int = 32
    enable_gpu: bool = False
    debug: bool = False

    def validate(self) -> bool:
        for path in [self.data_raw_dir, self.data_processed_dir, self.data_embeddings_dir]:
            path.mkdir(parents=True, exist_ok=True)
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not set — LLM generation features disabled")
        return True


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
        _config.validate()
    return _config
