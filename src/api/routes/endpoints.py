#!/usr/bin/env python3
"""
API route handlers
"""

from fastapi import APIRouter, HTTPException
from src.api.routes.schemas import (
    QueryRequest,
    AnswerResponse,
    HealthResponse,
    StatsResponse,
    IndexRequest,
    IndexResponse,
    Citation,
    ClassifyRequest,
    ClassifyResponse,
)
from src.generation.rag_pipeline_new import new_rag_pipeline
from src.retrieval.vector_store import vector_store
from src.retrieval.langchain_retriever import langchain_retriever
from src.classification.feedback_classifier import feedback_classifier
from src.config.settings import config
from src.utils.logger import logger

router = APIRouter()


@router.post("/answer", response_model=AnswerResponse)
async def answer_endpoint(request: QueryRequest):
    """
    Main answer endpoint for RAG (NEW)

    Process a user query through the complete RAG pipeline with LangChain
    and return an answer with citations, steps, warnings, and metrics.
    """
    try:
        result = await new_rag_pipeline.process_query(
            query=request.query,
            language=request.language,
            user_filters=request.filters,
            skip_query_analysis=request.skip_query_analysis,
        )

        # Convert citations to Citation objects
        citations = []
        for c in result["citations"]:
            # Safely convert page_num to int, handle invalid values
            page_num = c.get("page_num")
            if page_num is not None:
                try:
                    page_num = int(page_num) if page_num else None
                except (ValueError, TypeError):
                    page_num = None  # Set to None if conversion fails

            citations.append(
                Citation(
                    doc_id=c.get("doc_id", ""),
                    section=c.get("section", ""),
                    page_num=page_num,
                    chunk_id=c.get("chunk_id"),
                )
            )

        return AnswerResponse(
            answer=result["answer"],
            citations=citations,
            steps=result.get("steps", []),
            warnings=result.get("warnings", []),
            confidence=result["confidence"],
            latency_ms=result["latency_ms"],
            token_cost_estimate=result["token_cost_estimate"],
        )

    except Exception as e:
        logger.error(f"Error in answer endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns the status of all pipeline components and vector store.
    """
    try:
        # Get pipeline status
        pipeline_status = new_rag_pipeline.get_pipeline_status()

        vector_store_count = vector_store.get_collection_count()

        # Add classifier status
        pipeline_status["classifier"] = (
            "ready" if feedback_classifier.is_ready() else "not_loaded"
        )

        # Determine overall status
        all_ready = all(
            status in ["ready", "unavailable", "not_loaded"]
            for status in pipeline_status.values()
        )
        overall_status = (
            "healthy" if all_ready and vector_store_count > 0 else "degraded"
        )

        return HealthResponse(
            status=overall_status,
            components=pipeline_status,
            vector_store_count=vector_store_count,
        )

    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get vector store statistics

    Returns information about the vector database.
    """
    try:
        total_chunks = vector_store.get_collection_count()

        return StatsResponse(
            total_chunks=total_chunks,
            collection_name=config.QDRANT_COLLECTION_NAME,
            qdrant_host=config.QDRANT_HOST,
            qdrant_port=config.QDRANT_PORT,
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index", response_model=IndexResponse)
async def trigger_indexing(request: IndexRequest):
    """
    Manually trigger indexing

    Admin endpoint to reindex all chunks into the vector store.
    """
    try:
        logger.info("Manual indexing triggered")

        # Check if force reindex
        if request.force_reindex:
            logger.info("Force reindex requested - recreating collection")
            if vector_store.collection_exists():
                vector_store.client.delete_collection(config.QDRANT_COLLECTION_NAME)
            vector_store.create_collection()

        # Load and index chunks
        chunks = vector_store.load_chunks_from_directory()

        if not chunks:
            return IndexResponse(
                status="warning", message="No chunks found to index", chunks_indexed=0
            )

        vector_store.index_chunks(chunks)

        # Initialize LangChain retriever
        try:
            langchain_retriever.initialize_bm25(chunks)
            langchain_retriever.initialize_vector_store()
            logger.info("LangChain retriever initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LangChain retriever: {e}")

        return IndexResponse(
            status="success",
            message=f"Successfully indexed {len(chunks)} chunks",
            chunks_indexed=len(chunks),
        )

    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify", response_model=ClassifyResponse)
async def classify_feedback(request: ClassifyRequest):
    """
    Classify customer feedback

    Classifies feedback text into:
    - Level 1: Technical, Payment, or Claims
    - Level 2: 8 subcategories (Login, App_Performance, Refund, Limit, etc.)

    Returns predicted categories with confidence scores.
    """
    try:
        # Validate language
        if request.language not in ["en", "ar"]:
            raise HTTPException(status_code=400, detail="Language must be 'en' or 'ar'")

        # Check if classifier is ready
        if not feedback_classifier.is_ready():
            try:
                logger.info("Loading feedback classifier...")
                feedback_classifier.load_model()
            except FileNotFoundError as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Classifier model not available. Please train the model first. Details: {str(e)}",
                )

        # Classify the feedback
        result = feedback_classifier.classify(
            text=request.text, language=request.language
        )

        return ClassifyResponse(
            text=result["text"],
            language=result["language"],
            level1=result["level1"],
            level1_confidence=result["level1_confidence"],
            level2=result["level2"],
            level2_confidence=result["level2_confidence"],
        )

    except ValueError as e:
        logger.error(f"Validation error in classify endpoint: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error in classify endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Knowledge Assistant API",
        "version": "2.0.0",
        "endpoints": {
            "answer": "POST /answer - Submit a query",
            "classify": "POST /classify - Classify customer feedback",
            "health": "GET /health - Check system health",
            "stats": "GET /stats - Get vector store statistics",
            "index": "POST /index - Trigger manual indexing",
            "docs": "GET /docs - Interactive API documentation",
        },
    }
