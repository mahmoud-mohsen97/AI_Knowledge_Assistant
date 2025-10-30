#!/usr/bin/env python3
"""
FastAPI application setup
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.endpoints import router
from src.retrieval.vector_store import vector_store
from src.retrieval.langchain_retriever import langchain_retriever
from src.classification.feedback_classifier import feedback_classifier
from src.utils.logger import logger, setup_logger
from src.config.settings import config

# Setup logger
setup_logger(level=config.LOG_LEVEL)

# Create FastAPI app
app = FastAPI(
    title="AI Knowledge Assistant",
    description="RAG-based question answering system for insurance knowledge base",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler

    Initialize vector store and BM25 handler on application startup.
    """
    logger.info("=" * 80)
    logger.info("AI Knowledge Assistant Starting Up")
    logger.info("=" * 80)

    try:
        # Validate configuration
        logger.info("Validating configuration...")
        config.validate()
        logger.info("Configuration validated")

        # Initialize vector store (auto-index if empty)
        logger.info("Initializing vector store...")
        vector_store.initialize_and_index_if_empty()
        logger.info("Vector store initialized")

        # Load chunks once for all initializations
        logger.info("Loading chunks...")
        chunks = vector_store.load_chunks_from_directory()
        if not chunks:
            logger.warning("⚠️  No chunks found")
        else:
            logger.info(f"✓ Loaded {len(chunks)} chunks")

        # Initialize LangChain retriever
        logger.info("Initializing LangChain retriever...")
        try:
            if chunks:
                langchain_retriever.initialize_bm25(chunks)
                langchain_retriever.initialize_vector_store()
                logger.info("✓ LangChain retriever initialized")
            else:
                logger.warning("⚠️  Skipping retriever initialization (no chunks)")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize LangChain retriever: {e}")

        # Initialize feedback classifier
        logger.info("Initializing feedback classifier...")
        try:
            feedback_classifier.load_model()
            logger.info("✓ Feedback classifier loaded and ready")
        except FileNotFoundError:
            logger.warning("⚠️  Feedback classifier model not found")
            logger.warning(
                "   Train the model using: notebooks/feedback_classification_multitask.ipynb"
            )
        except Exception as e:
            logger.warning(f"⚠️  Could not load feedback classifier: {e}")

        logger.info("=" * 80)
        logger.info("AI Knowledge Assistant Ready")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("AI Knowledge Assistant shutting down...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower(),
    )
