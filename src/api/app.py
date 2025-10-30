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
        # Validate configuration (skip chunks check initially)
        logger.info("Validating configuration...")
        config.validate(skip_chunks_check=True)
        logger.info("Configuration validated")

        # Check if chunks directory exists and has data
        if not config.PROCESSED_CHUNKS_DIR.exists() or not any(
            config.PROCESSED_CHUNKS_DIR.glob("*.json")
        ):
            logger.warning("Chunks directory not found or empty")
            logger.info("Running ingestion pipeline to create chunks...")

            try:
                # Import and run the ingestion pipeline
                from src.ingestion.pipeline import process_all_documents

                # Ensure necessary directories exist
                config.create_directories()

                # Run ingestion pipeline
                process_all_documents(
                    json_dir="data/processed/pdf_extracts",
                    image_dir="data/processed/pdf_images",
                    output_dir="data/processed/visual_chunks",
                    pdf_dir="data/raw/docs",
                    extract_first=True,
                    verbose=True,
                    create_final_chunks=True,
                    faq_path="data/raw/faq.json",
                    tickets_path="data/raw/tickets_resolved.txt",
                    doc_metadata_path="data/raw/doc_metadata.json",
                    all_chunks_dir="data/processed/all_chunks",
                )
                logger.info("✓ Ingestion pipeline completed successfully")
            except Exception as e:
                logger.error(f"Error running ingestion pipeline: {e}", exc_info=True)
                logger.warning("Continuing startup without chunks...")
        else:
            chunk_count = len(list(config.PROCESSED_CHUNKS_DIR.glob("*.json")))
            logger.info(f"✓ Found {chunk_count} chunk files")

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
