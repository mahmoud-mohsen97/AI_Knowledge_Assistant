#!/usr/bin/env python3
"""
Hybrid retrieval system combining BM25 and dense vector search
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from src.config.settings import config
from src.retrieval.vector_store import vector_store
from src.retrieval.bm25_handler import bm25_handler
from src.retrieval.embedder import embedding_generator
from src.utils.logger import logger


class HybridRetriever:
    """Hybrid retriever combining BM25 (sparse) and dense vector search"""

    def __init__(self):
        """Initialize hybrid retriever"""
        self.vector_store = vector_store
        self.bm25_handler = bm25_handler
        self.embedding_generator = embedding_generator
        self.rrf_k = config.RRF_K
        logger.info("Initialized HybridRetriever")

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[tuple],
        k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion

        Args:
            dense_results: Results from dense retrieval (from Qdrant)
            sparse_results: Results from BM25 as (index, score) tuples
            k: RRF constant (default: 60)

        Returns:
            Fused results sorted by RRF score
        """
        k = k or self.rrf_k
        rrf_scores = defaultdict(float)
        chunk_data = {}

        # Process dense results
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result["payload"].get("chunk_id")
            if chunk_id:
                rrf_scores[chunk_id] += 1.0 / (k + rank)
                chunk_data[chunk_id] = result

        # Process sparse (BM25) results
        for rank, (idx, score) in enumerate(sparse_results, start=1):
            chunk = self.bm25_handler.get_chunk_by_index(idx)
            chunk_id = chunk.get("chunk_id")

            if chunk_id:
                rrf_scores[chunk_id] += 1.0 / (k + rank)

                # If not in chunk_data, add it
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = {"id": idx, "score": score, "payload": chunk}

        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return results with RRF scores
        results = []
        for chunk_id, rrf_score in sorted_chunks:
            if chunk_id in chunk_data:
                result = chunk_data[chunk_id].copy()
                result["rrf_score"] = rrf_score
                results.append(result)

        logger.debug(f"RRF fused {len(results)} unique results")
        return results

    def apply_query_boosting(
        self,
        results: List[Dict[str, Any]],
        query_intent: str = None,
        chunk_type_preference: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply boosting based on query intent and chunk type

        Args:
            results: Retrieved results
            query_intent: Detected query intent
            chunk_type_preference: Preferred chunk type

        Returns:
            Results with adjusted scores
        """
        if not query_intent and not chunk_type_preference:
            return results

        # Define boosting rules
        boost_rules = {
            "process_flow": {"flowchart": 1.5, "visual": 1.3},
            "visual_interpretation": {"flowchart": 1.5, "chart": 1.5, "visual": 1.3},
            "numerical_query": {"chart": 1.5, "table": 1.3},
            "policy_lookup": {"text": 1.2},
            "procedural": {"text": 1.2, "flowchart": 1.3},
        }

        # Get boost factors
        boost_factors = {}
        if query_intent and query_intent in boost_rules:
            boost_factors.update(boost_rules[query_intent])

        if chunk_type_preference:
            boost_factors[chunk_type_preference] = (
                boost_factors.get(chunk_type_preference, 1.0) * 1.3
            )

        # Apply boosting
        for result in results:
            chunk_type = result["payload"].get("chunk_type", "text")
            if chunk_type in boost_factors:
                boost = boost_factors[chunk_type]
                result["rrf_score"] = (
                    result.get("rrf_score", result.get("score", 1.0)) * boost
                )
                logger.debug(f"Applied boost {boost} to chunk type {chunk_type}")

        # Re-sort by adjusted score
        results.sort(key=lambda x: x.get("rrf_score", x.get("score", 0)), reverse=True)

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        query_intent: str = None,
        chunk_type_preference: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse retrieval

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            query_intent: Detected query intent for boosting
            chunk_type_preference: Preferred chunk type

        Returns:
            List of retrieved chunks with scores
        """
        top_k = top_k or config.RETRIEVAL_TOP_K

        logger.info(f"Hybrid search for query: '{query[:100]}...'")

        # 1. Dense retrieval (vector search)
        logger.debug("Performing dense retrieval...")
        query_embedding = self.embedding_generator.generate_embedding(query)

        qdrant_filter = self.vector_store.build_filter(filters) if filters else None

        dense_results = self.vector_store.search(
            query_vector=query_embedding, limit=top_k, filters=qdrant_filter
        )

        logger.info(f"Dense retrieval returned {len(dense_results)} results")

        # 2. Sparse retrieval (BM25)
        sparse_results = []
        if self.bm25_handler.is_ready():
            logger.debug("Performing BM25 retrieval...")
            sparse_results = self.bm25_handler.search(query, top_k=top_k)
            logger.info(f"BM25 retrieval returned {len(sparse_results)} results")
        else:
            logger.warning("BM25 handler not ready, skipping sparse retrieval")

        # 3. Reciprocal Rank Fusion
        logger.debug("Applying Reciprocal Rank Fusion...")
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)

        # 4. Apply query-aware boosting
        if query_intent or chunk_type_preference:
            logger.debug("Applying query-aware boosting...")
            fused_results = self.apply_query_boosting(
                fused_results,
                query_intent=query_intent,
                chunk_type_preference=chunk_type_preference,
            )

        # 5. Return top-k
        final_results = fused_results[:top_k]
        logger.info(f"Hybrid search returning {len(final_results)} results")

        return final_results

    def initialize_bm25(self):
        """Initialize BM25 index from vector store chunks"""
        if self.bm25_handler.is_ready():
            logger.info("BM25 handler already initialized")
            return

        logger.info("Initializing BM25 handler...")
        chunks = self.vector_store.load_chunks_from_directory()

        if chunks:
            self.bm25_handler.build_index(chunks)
            logger.info("BM25 handler initialized successfully")
        else:
            logger.warning("No chunks available for BM25 initialization")


# Singleton instance
hybrid_retriever = HybridRetriever()
