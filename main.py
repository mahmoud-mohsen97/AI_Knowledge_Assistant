#!/usr/bin/env python3
"""
Main entry point for AI Knowledge Assistant
Starts the FastAPI server with RAG pipeline
"""

import uvicorn
from src.config.settings import config


def main():
    """Start the FastAPI application"""
    print("=" * 80)
    print("Starting AI Knowledge Assistant")
    print(f"API will be available at http://{config.API_HOST}:{config.API_PORT}")
    print(f"Interactive docs at http://{config.API_HOST}:{config.API_PORT}/docs")
    print("=" * 80)

    uvicorn.run(
        "src.api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
