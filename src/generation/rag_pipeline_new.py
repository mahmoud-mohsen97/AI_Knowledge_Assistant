#!/usr/bin/env python3
"""
End-to-end RAG pipeline orchestration with LangChain
"""

import time
from typing import Dict, Any, Optional
from src.generation.query_analyzer import query_analyzer
from src.retrieval.langchain_retriever import langchain_retriever
from src.retrieval.jina_reranker import jina_reranker
from src.generation.context_builder import context_builder
from src.generation.answer_generator import answer_generator
from src.utils.logger import logger


class NewRAGPipeline:
    """Orchestrate the complete RAG pipeline with LangChain integration"""

    def __init__(self):
        """Initialize RAG pipeline with all components"""
        self.query_analyzer = query_analyzer
        self.retriever = langchain_retriever
        self.reranker = jina_reranker
        self.context_builder = context_builder
        self.answer_generator = answer_generator
        logger.info("Initialized NewRAGPipeline with LangChain")

    async def process_query(
        self,
        query: str,
        language: str = "English",
        user_filters: Optional[Dict[str, Any]] = None,
        skip_query_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline

        Args:
            query: User query
            language: Response language ("English" or "Arabic")
            user_filters: Optional user-provided filters
            skip_query_analysis: Skip LLM-based query analysis

        Returns:
            Dictionary with answer, citations, steps, warnings, and metrics
        """
        start_time = time.time()
        logger.info(f"Processing query: '{query[:100]}...' in {language}")

        result = {
            "answer": "",
            "citations": [],
            "steps": [],
            "warnings": [],
            "confidence": 0.0,
            "latency_ms": 0,
            "token_cost_estimate": 0,
        }

        try:
            # Convert language to ISO code
            lang_code = "ar" if language.lower() == "arabic" else "en"

            # Step 1: Query Analysis
            filters = user_filters or {}
            if not skip_query_analysis:
                logger.info("[Step 1/5] Analyzing query...")
                analysis = self.query_analyzer.analyze(query)

                # Build filters from analysis
                if analysis.doc_type:
                    filters["doc_type"] = analysis.doc_type
                if analysis.section_title:
                    filters["section_title"] = analysis.section_title
                if analysis.chunk_type:
                    filters["chunk_type"] = analysis.chunk_type

                # Merge with user filters
                filters.update(user_filters or {})

                logger.info(
                    f"Query analysis: intent={analysis.intent}, doc_type={analysis.doc_type}, filters={filters}"
                )
            else:
                logger.info("[Step 1/5] Skipping query analysis")
                analysis = None

            # Step 2: Unified Vector Search with Metadata Filters
            logger.info("[Step 2/5] Performing hybrid retrieval with LangChain...")
            retrieved_chunks = self.retriever.search(
                query=query,
                filters=filters,
                top_k=20,  # Retrieve more for reranking
            )

            if not retrieved_chunks:
                logger.warning("No chunks retrieved")
                result["answer"] = (
                    "I couldn't find any relevant information to answer your question."
                )
                result["confidence"] = 0.0
                result["latency_ms"] = int((time.time() - start_time) * 1000)
                return result

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

            # Step 3: Jina Reranker
            logger.info("[Step 3/5] Reranking with Jina AI...")
            if self.reranker.is_available():
                reranked_chunks = self.reranker.rerank(query, retrieved_chunks, top_n=5)
                rerank_scores = [c.get("rerank_score", 0.0) for c in reranked_chunks]
                logger.info(f"Reranked to top {len(reranked_chunks)} chunks")
            else:
                logger.warning("Jina reranker not available, using retrieval results")
                reranked_chunks = retrieved_chunks[:5]
                rerank_scores = []

            # Step 4: Context Assembly
            logger.info("[Step 4/5] Assembling context...")
            context_data = self.context_builder.build_context_with_metadata(
                reranked_chunks
            )
            formatted_context = context_data["formatted_context"]
            citations_list = context_data["citations"]
            extracted_warnings = context_data["warnings"]
            extracted_steps = context_data["steps"]

            logger.info(
                f"Context: {len(citations_list)} citations, {len(extracted_steps)} steps, {len(extracted_warnings)} warnings"
            )

            # Step 5: Answer Generation
            logger.info("[Step 5/5] Generating answer...")
            answer = self.answer_generator.generate(
                query=query,
                context=formatted_context,
                language=lang_code,
                citations_list=citations_list,
                extracted_steps=extracted_steps,
                extracted_warnings=extracted_warnings,
                rerank_scores=rerank_scores,
            )

            # Calculate total latency
            total_latency_ms = int((time.time() - start_time) * 1000)

            # Update answer with total latency
            answer.latency_ms = total_latency_ms

            # Convert citations to new format (section instead of section_title)
            formatted_citations = []
            for citation in answer.citations:
                formatted_citations.append(
                    {
                        "doc_id": citation.get("doc_id", ""),
                        "section": citation.get(
                            "section_title", citation.get("section", "")
                        ),
                        "page_num": citation.get("page_num"),
                        "chunk_id": citation.get("chunk_id"),
                    }
                )

            result = {
                "answer": answer.answer,
                "citations": formatted_citations,
                "steps": answer.steps,
                "warnings": answer.warnings,
                "confidence": answer.confidence,
                "latency_ms": answer.latency_ms,
                "token_cost_estimate": answer.token_cost_estimate,
            }

            logger.info(
                f"Pipeline complete: {answer.latency_ms}ms, confidence={answer.confidence}"
            )

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
            result["answer"] = (
                "I encountered an error while processing your question. Please try again."
            )
            result["confidence"] = 0.0
            result["latency_ms"] = int((time.time() - start_time) * 1000)

        return result

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get status of all pipeline components

        Returns:
            Dictionary with component statuses
        """
        return {
            "query_analyzer": "ready",
            "langchain_retriever": "ready"
            if self.retriever.is_ready()
            else "not initialized",
            "jina_reranker": "ready" if self.reranker.is_available() else "unavailable",
            "context_builder": "ready",
            "answer_generator": "ready",
        }


# Singleton instance
new_rag_pipeline = NewRAGPipeline()
