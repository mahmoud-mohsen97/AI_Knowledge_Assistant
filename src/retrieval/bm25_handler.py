#!/usr/bin/env python3
"""
BM25 implementation for keyword-based retrieval
"""

from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from src.utils.logger import logger


# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt", quiet=True)


class BM25Handler:
    """BM25 keyword-based retrieval handler"""

    def __init__(self):
        """Initialize BM25 handler"""
        self.bm25 = None
        self.corpus_chunks = []
        self.tokenized_corpus = []
        logger.info("Initialized BM25Handler")

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Build BM25 index from chunks

        Args:
            chunks: List of chunk dictionaries with 'content' field
        """
        if not chunks:
            logger.warning("No chunks provided for BM25 indexing")
            return

        logger.info(f"Building BM25 index for {len(chunks)} chunks")

        self.corpus_chunks = chunks

        # Tokenize corpus
        self.tokenized_corpus = []
        for chunk in chunks:
            content = chunk.get("content", "")
            if content:
                tokens = self._tokenize(content)
                self.tokenized_corpus.append(tokens)
            else:
                self.tokenized_corpus.append([])

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built successfully")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text:
            return []

        try:
            tokens = word_tokenize(text.lower())
            # Filter out very short tokens
            tokens = [t for t in tokens if len(t) > 1]
            return tokens
        except Exception as e:
            logger.warning(f"Error tokenizing text: {e}")
            return text.lower().split()

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Search using BM25

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (chunk_index, score) tuples sorted by score
        """
        if not self.bm25:
            logger.warning("BM25 index not built")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            logger.warning("Empty tokenized query")
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results = [(idx, scores[idx]) for idx in top_indices]

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    def get_chunk_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get chunk by index

        Args:
            index: Chunk index

        Returns:
            Chunk dictionary
        """
        if 0 <= index < len(self.corpus_chunks):
            return self.corpus_chunks[index]
        return {}

    def is_ready(self) -> bool:
        """Check if BM25 index is ready"""
        return self.bm25 is not None and len(self.corpus_chunks) > 0


# Singleton instance
bm25_handler = BM25Handler()
