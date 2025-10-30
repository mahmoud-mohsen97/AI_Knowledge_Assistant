#!/usr/bin/env python3
"""
Reranking module using Jina Reranker
"""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import torch
from src.config.settings import config
from src.utils.logger import logger


class Reranker:
    """Rerank retrieved chunks using cross-encoder model"""

    def __init__(self, model_name: str = None):
        """
        Initialize reranker

        Args:
            model_name: Reranker model name
        """
        self.model_name = model_name or config.RERANKER_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading reranker model: {self.model_name} on {self.device}")

        try:
            self.model = CrossEncoder(
                self.model_name, device=self.device, trust_remote_code=True
            )
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            self.model = None

    def rerank(
        self, query: str, chunks: List[Dict[str, Any]], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on query relevance

        Args:
            query: Search query
            chunks: Retrieved chunks to rerank
            top_k: Number of top results to return

        Returns:
            Reranked chunks with scores
        """
        if not self.model:
            logger.warning("Reranker model not available, returning original chunks")
            return chunks[:top_k] if top_k else chunks

        if not chunks:
            return []

        top_k = top_k or config.RERANKER_TOP_K

        logger.info(f"Reranking {len(chunks)} chunks")

        try:
            # Prepare query-chunk pairs
            pairs = []
            for chunk in chunks:
                content = chunk["payload"].get("content", "")
                pairs.append([query, content])

            # Get reranking scores
            scores = self.model.predict(pairs)

            # Attach scores to chunks
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)

            # Sort by rerank score
            reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

            # Return top-k
            result = reranked[:top_k]

            logger.info(f"Reranking complete, returning top {len(result)} results")
            return result

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return chunks[:top_k]

    def is_available(self) -> bool:
        """Check if reranker is available"""
        return self.model is not None


# Singleton instance
reranker = Reranker()
