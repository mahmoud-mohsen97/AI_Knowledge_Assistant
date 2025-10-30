#!/usr/bin/env python3
"""
End-to-end RAG pipeline orchestration
"""

from typing import Dict, Any, Optional
from src.generation.query_analyzer import query_analyzer
from src.retrieval.hybrid_retriever import hybrid_retriever
from src.retrieval.reranker import reranker
from src.generation.context_builder import context_builder
from src.generation.answer_generator import answer_generator
from src.utils.logger import logger


class RAGPipeline:
    """Orchestrate the complete RAG pipeline"""

    def __init__(self):
        """Initialize RAG pipeline with all components"""
        self.query_analyzer = query_analyzer
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.answer_generator = answer_generator
        logger.info("Initialized RAGPipeline")

    async def process_query(
        self,
        query: str,
        user_filters: Optional[Dict[str, Any]] = None,
        skip_query_analysis: bool = False,
        use_few_shot: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline

        Args:
            query: User query
            user_filters: Optional user-provided filters
            skip_query_analysis: Skip LLM-based query analysis
            use_few_shot: Use few-shot prompting for answer generation

        Returns:
            Dictionary with answer, citations, metadata
        """
        logger.info(f"Processing query: '{query[:100]}...'")

        result = {
            "query": query,
            "answer": "",
            "citations": [],
            "confidence": "low",
            "language": "en",
            "metadata": {},
        }

        try:
            # Step 1: Query Analysis
            if not skip_query_analysis:
                logger.info("[Step 1/5] Analyzing query...")
                analysis = self.query_analyzer.analyze(query)
                result["metadata"]["query_analysis"] = {
                    "intent": analysis.intent,
                    "language": analysis.language,
                    "detected_filters": analysis.filters,
                    "chunk_type_preference": analysis.chunk_type_preference,
                    "analysis_confidence": analysis.confidence,
                }
                language = analysis.language
                intent = analysis.intent
                chunk_type_pref = analysis.chunk_type_preference

                # Merge user filters with detected filters
                filters = {**analysis.filters, **(user_filters or {})}
            else:
                logger.info("[Step 1/5] Skipping query analysis")
                language = "en"
                intent = None
                chunk_type_pref = None
                filters = user_filters or {}

            result["language"] = language

            # Step 2: Hybrid Retrieval
            logger.info("[Step 2/5] Performing hybrid retrieval...")
            retrieved_chunks = self.retriever.hybrid_search(
                query=query,
                filters=filters,
                query_intent=intent,
                chunk_type_preference=chunk_type_pref,
            )

            result["metadata"]["retrieval"] = {
                "num_chunks_retrieved": len(retrieved_chunks),
                "filters_applied": filters,
            }

            if not retrieved_chunks:
                logger.warning("No chunks retrieved")
                result["answer"] = (
                    "I couldn't find any relevant information to answer your question."
                )
                result["confidence"] = "low"
                return result

            # Step 3: Reranking
            logger.info("[Step 3/5] Reranking results...")
            if self.reranker.is_available():
                reranked_chunks = self.reranker.rerank(query, retrieved_chunks)
                result["metadata"]["reranking"] = {
                    "reranker_used": True,
                    "num_chunks_after_rerank": len(reranked_chunks),
                }
            else:
                logger.warning("Reranker not available, using retrieval results")
                reranked_chunks = retrieved_chunks[:5]  # Take top 5
                result["metadata"]["reranking"] = {
                    "reranker_used": False,
                    "num_chunks_after_rerank": len(reranked_chunks),
                }

            # Step 4: Context Assembly
            logger.info("[Step 4/5] Assembling context...")
            context_data = self.context_builder.build_context_with_metadata(
                reranked_chunks
            )
            formatted_context = context_data["formatted_context"]
            citations_list = context_data["citations"]

            result["metadata"]["context"] = {
                "num_chunks_used": context_data["num_chunks"],
                "chunk_types": context_data["chunk_types"],
            }

            # Step 5: Answer Generation
            logger.info("[Step 5/5] Generating answer...")
            if use_few_shot:
                answer = self.answer_generator.generate_with_few_shot(
                    query=query, context=formatted_context, language=language
                )
            else:
                answer = self.answer_generator.generate(
                    query=query,
                    context=formatted_context,
                    language=language,
                    citations_list=citations_list,
                )

            result["answer"] = answer.answer
            result["citations"] = answer.citations
            result["confidence"] = answer.confidence

            logger.info(
                f"Pipeline complete: {len(answer.answer)} chars, {len(answer.citations)} citations"
            )

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
            result["answer"] = (
                "I encountered an error while processing your question. Please try again."
            )
            result["confidence"] = "low"
            result["metadata"]["error"] = str(e)

        return result

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get status of all pipeline components

        Returns:
            Dictionary with component statuses
        """
        return {
            "query_analyzer": "ready",
            "retriever": "ready",
            "reranker": "ready" if self.reranker.is_available() else "unavailable",
            "context_builder": "ready",
            "answer_generator": "ready",
            "bm25_handler": "ready"
            if self.retriever.bm25_handler.is_ready()
            else "not initialized",
        }


# Singleton instance
rag_pipeline = RAGPipeline()
