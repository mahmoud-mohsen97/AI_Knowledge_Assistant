#!/usr/bin/env python3
"""
Configuration management for the AI Knowledge Assistant
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration class"""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_RAW_DIR = DATA_DIR / "raw"
    DATA_PROCESSED_DIR = DATA_DIR / "processed"

    # Data subdirectories
    RAW_DOCS_DIR = DATA_RAW_DIR / "docs"
    RAW_FAQ_PATH = DATA_RAW_DIR / "faq.json"
    RAW_TICKETS_PATH = DATA_RAW_DIR / "tickets_resolved.txt"
    RAW_METADATA_PATH = DATA_RAW_DIR / "doc_metadata.json"

    PROCESSED_CHUNKS_DIR = DATA_PROCESSED_DIR / "all_chunks"
    PROCESSED_PDF_EXTRACTS_DIR = DATA_PROCESSED_DIR / "pdf_extracts"
    PROCESSED_IMAGES_DIR = DATA_PROCESSED_DIR / "pdf_images"
    PROCESSED_VISUAL_CHUNKS_DIR = DATA_PROCESSED_DIR / "visual_chunks"

    # Models directory
    MODELS_DIR = PROJECT_ROOT / "models"
    EMBEDDINGS_CACHE_DIR = MODELS_DIR / "embeddings" / "cache"

    # Logs and outputs
    LOGS_DIR = PROJECT_ROOT / "logs"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Jina AI Configuration
    JINA_API_KEY: str = os.getenv(
        "JINA_API_KEY",
        "jina_9b637d292b934b9a92dd05ac6d63f2b4A5fL1wVg7sjE5aAkfx2aJ76UrMX4",
    )

    # Qdrant Configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = os.getenv(
        "QDRANT_COLLECTION_NAME", "insurance_knowledge_base"
    )
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual"
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5-nano")

    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "5"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls, skip_chunks_check: bool = False) -> bool:
        """Validate required configuration

        Args:
            skip_chunks_check: Skip validation of chunks directory (used during startup before ingestion)
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")

        if not skip_chunks_check and not cls.PROCESSED_CHUNKS_DIR.exists():
            raise ValueError(f"Chunks directory not found: {cls.PROCESSED_CHUNKS_DIR}")

        return True

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_RAW_DIR,
            cls.DATA_PROCESSED_DIR,
            cls.RAW_DOCS_DIR,
            cls.PROCESSED_CHUNKS_DIR,
            cls.PROCESSED_PDF_EXTRACTS_DIR,
            cls.PROCESSED_IMAGES_DIR,
            cls.PROCESSED_VISUAL_CHUNKS_DIR,
            cls.MODELS_DIR,
            cls.EMBEDDINGS_CACHE_DIR,
            cls.LOGS_DIR,
            cls.OUTPUTS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Create a singleton instance
config = Config()
