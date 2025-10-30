#!/usr/bin/env python3
"""
Reranking module using Jina AI API
"""

from typing import List, Dict, Any
import requests
from src.config.settings import config
from src.utils.logger import logger


class JinaReranker:
    """Rerank retrieved chunks using Jina AI Reranker API"""

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Jina reranker

        Args:
            api_key: Jina AI API key
            model: Reranker model name
        """
        self.api_key = api_key or config.JINA_API_KEY
        self.model = model or "jina-reranker-v2-base-multilingual"
        self.api_url = "https://api.jina.ai/v1/rerank"
        logger.info(f"Initialized JinaReranker with model: {self.model}")

    def rerank(
        self, query: str, chunks: List[Dict[str, Any]], top_n: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on query relevance using Jina AI API

        Args:
            query: Search query
            chunks: Retrieved chunks to rerank
            top_n: Number of top results to return

        Returns:
            Reranked chunks with scores
        """
        if not chunks:
            return []

        top_n = top_n or config.RERANKER_TOP_K

        logger.info(f"Reranking {len(chunks)} chunks using Jina AI API")

        try:
            # Prepare documents for reranking
            documents = []
            for chunk in chunks:
                # Extract content from different chunk structures
                if "payload" in chunk:
                    content = chunk["payload"].get("content", "")
                    # For FAQ and ticket chunks, combine question and answer
                    chunk_type = chunk["payload"].get("chunk_type", "")
                    if chunk_type == "faq" or chunk_type == "resolved_ticket":
                        question = chunk["payload"].get("question", "")
                        answer = chunk["payload"].get("answer", "")
                        content = f"{question} {answer}".strip()
                else:
                    content = chunk.get("content", "")

                documents.append(content)

            # Make API request to Jina
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            data = {
                "model": self.model,
                "query": query,
                "top_n": min(top_n, len(chunks)),  # Don't request more than we have
                "documents": documents,
                "return_documents": False,  # We already have the documents
            }

            response = requests.post(
                self.api_url, headers=headers, json=data, timeout=30
            )
            response.raise_for_status()

            result = response.json()

            # Map reranked results back to original chunks
            reranked_chunks = []
            for item in result.get("results", []):
                index = item.get("index")
                score = item.get("relevance_score", 0.0)

                if index is not None and index < len(chunks):
                    chunk = chunks[index].copy()
                    chunk["rerank_score"] = float(score)
                    reranked_chunks.append(chunk)

            logger.info(
                f"Reranking complete, returning top {len(reranked_chunks)} results"
            )
            return reranked_chunks

        except requests.exceptions.RequestException as e:
            logger.error(f"Jina API request failed: {e}")
            logger.warning("Falling back to original retrieval order")
            # Fallback: return top_n chunks without reranking
            return chunks[:top_n]

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            logger.warning("Falling back to original retrieval order")
            return chunks[:top_n]

    def is_available(self) -> bool:
        """
        Check if Jina reranker API is available

        Returns:
            True if API key is configured
        """
        return bool(self.api_key)


# Singleton instance
jina_reranker = JinaReranker()
