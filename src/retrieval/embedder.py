#!/usr/bin/env python3
"""
Embedding generation utilities using OpenAI's text-embedding-3-large
"""

from typing import Any, Dict, List
from openai import OpenAI
from src.config.settings import config
from src.utils.logger import logger


class EmbeddingGenerator:
    """Generate embeddings using OpenAI API"""

    def __init__(self, model: str = None):
        """
        Initialize embedding generator

        Args:
            model: OpenAI embedding model name
        """
        self.model = model or config.EMBEDDING_MODEL
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.dimension = config.EMBEDDING_DIMENSION
        logger.info(f"Initialized EmbeddingGenerator with model: {self.model}")

    def prepare_text_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """
        Prepare text for embedding based on chunk type

        Strategy:
        - Document chunks: content + doc_title + section_title
        - FAQ chunks: question + answer
        - Resolved ticket chunks: question + answer

        Args:
            chunk: Chunk dictionary with metadata

        Returns:
            Combined text ready for embedding
        """
        chunk_type = chunk.get("chunk_type", "")

        # Handle FAQ chunks
        if chunk_type == "faq":
            question = chunk.get("question", "")
            answer = chunk.get("answer", "")
            text = f"{question} {answer}".strip()
            logger.debug(f"Prepared FAQ chunk for embedding: {len(text)} chars")
            return text

        # Handle resolved ticket chunks
        elif chunk_type == "resolved_ticket":
            question = chunk.get("question", "")
            answer = chunk.get("answer", "")
            text = f"{question} {answer}".strip()
            logger.debug(f"Prepared ticket chunk for embedding: {len(text)} chars")
            return text

        # Handle document chunks (including flowcharts, charts, etc.)
        else:
            parts = []

            # Add document title
            doc_title = chunk.get("doc_title", "")
            if doc_title:
                parts.append(doc_title)

            # Add section title
            section_title = chunk.get("section_title", "")
            if section_title:
                parts.append(section_title)

            # Add main content
            content = chunk.get("content", "")
            if content:
                parts.append(content)

            text = " ".join(parts).strip()
            logger.debug(f"Prepared document chunk for embedding: {len(text)} chars")
            return text

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension

        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches

        Args:
            texts: List of input texts
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.embeddings.create(input=batch, model=self.model)
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                logger.info(
                    f"Generated embeddings for batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}"
                )
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                raise

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


# Singleton instance
embedding_generator = EmbeddingGenerator()
