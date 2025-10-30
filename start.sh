#!/bin/bash
# AI Knowledge Assistant 2.0 - Startup Script
# Supports both Docker and local development modes

set -e

echo "=========================================="
echo "AI Knowledge Assistant 2.0"
echo "=========================================="
echo ""

# Parse options
FORCE_REINDEX=false
RUN_TESTS=false
USE_DOCKER=false
BUILD_DOCKER=false
STOP_DOCKER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-reindex) FORCE_REINDEX=true; shift ;;
        --test) RUN_TESTS=true; shift ;;
        --docker) USE_DOCKER=true; shift ;;
        --build) BUILD_DOCKER=true; USE_DOCKER=true; shift ;;
        --down) STOP_DOCKER=true; shift ;;
        --help) 
            echo "Usage: ./start.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --docker          Run using Docker Compose"
            echo "  --build           Build Docker images and run"
            echo "  --down            Stop and remove Docker containers"
            echo "  --force-reindex   Force reindex of vector database"
            echo "  --test            Run tests before starting"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./start.sh                    # Run locally (default)"
            echo "  ./start.sh --docker           # Run with Docker"
            echo "  ./start.sh --build            # Build and run with Docker"
            echo "  ./start.sh --down             # Stop Docker containers"
            exit 0
            ;;
        *) echo "Unknown option: $1. Use --help for usage."; exit 1 ;;
    esac
done

# Handle Docker down
if [ "$STOP_DOCKER" = true ]; then
    echo "Stopping Docker containers..."
    docker-compose down
    echo "✓ Containers stopped"
    exit 0
fi

# 1. Check environment
echo "[1/5] Checking environment..."
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_api_key_here
JINA_API_KEY=jina_9b637d292b934b9a92dd05ac6d63f2b4A5fL1wVg7sjE5aAkfx2aJ76UrMX4
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=insurance_knowledge_base
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
EOF
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
    exit 1
fi

set -a; source .env; set +a

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "✗ OPENAI_API_KEY not set in .env"
    exit 1
fi
echo "✓ Environment configured"
echo ""

# Docker Mode
if [ "$USE_DOCKER" = true ]; then
    echo "=========================================="
    echo "Running in Docker Mode"
    echo "=========================================="
    echo ""
    
    # Build if requested
    if [ "$BUILD_DOCKER" = true ]; then
        echo "[2/5] Building Docker images..."
        docker-compose build
        echo "✓ Build complete"
        echo ""
    fi
    
    # Start services
    echo "[3/5] Starting Docker services..."
    docker-compose up -d
    echo "✓ Services started"
    echo ""
    
    # Wait for services to be healthy
    echo "[4/5] Waiting for services to be ready..."
    sleep 10
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        echo "✓ Services are running"
    else
        echo "✗ Failed to start services"
        docker-compose logs
        exit 1
    fi
    echo ""
    
    echo "=========================================="
    echo "Application Started (Docker)"
    echo "=========================================="
    echo "URL: http://localhost:8000"
    echo "Docs: http://localhost:8000/docs"
    echo "Qdrant Dashboard: http://localhost:6333/dashboard"
    echo ""
    echo "Endpoints:"
    echo "  POST /answer   - Knowledge Assistant Endpoint"
    echo "  POST /classify - Feedback classification endpoint"
    echo "  GET  /health   - Health check"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f          # All services"
    echo "  docker-compose logs -f api      # API only"
    echo "  docker-compose logs -f qdrant   # Qdrant only"
    echo ""
    echo "To stop:"
    echo "  ./start.sh --down"
    echo "  or"
    echo "  docker-compose down"
    echo ""
    
    exit 0
fi

# Local Development Mode
echo "=========================================="
echo "Running in Local Development Mode"
echo "=========================================="
echo ""
echo "Note: If you just ran ./fix_deps.sh, restart your terminal"
echo "      or run: source .venv/bin/activate"
echo ""

# 2. Start Qdrant
echo "[2/5] Starting Qdrant..."
if ! docker ps | grep -q qdrant 2>/dev/null; then
    if [ -f docker-compose.yml ]; then
        docker-compose up -d qdrant
    else
        docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
            -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
    fi
    sleep 5
    echo "✓ Qdrant started"
else
    echo "✓ Qdrant already running"
fi
echo ""

# 3. Check chunks
echo "[3/5] Checking data..."
CHUNKS_DIR="data/processed/all_chunks"
if [ ! -d "$CHUNKS_DIR" ] || [ -z "$(ls -A $CHUNKS_DIR 2>/dev/null)" ]; then
    echo "Running ingestion pipeline..."
    uv run -m src.ingestion.pipeline
    echo "✓ Ingestion complete"
else
    CHUNK_COUNT=$(ls -1 $CHUNKS_DIR/*.json 2>/dev/null | wc -l)
    echo "✓ Found $CHUNK_COUNT chunks"
fi
echo ""

# 4. Initialize RAG pipeline
echo "[4/5] Initializing RAG pipeline..."
if [ "$FORCE_REINDEX" = true ]; then
    echo "Force reindexing..."
    uv run python3 -c "
from src.retrieval.vector_store import vector_store
from src.retrieval.hybrid_retriever import hybrid_retriever
from src.retrieval.langchain_retriever import langchain_retriever
from src.config.settings import config

if vector_store.collection_exists():
    vector_store.client.delete_collection(config.QDRANT_COLLECTION_NAME)

vector_store.create_collection()
chunks = vector_store.load_chunks_from_directory()
vector_store.index_chunks(chunks)
hybrid_retriever.initialize_bm25()
langchain_retriever.initialize_bm25(chunks)
langchain_retriever.initialize_vector_store()
print('✓ Reindexing complete')
"
else
    # The API startup will handle initialization automatically
    echo "✓ Will initialize on startup"
fi
echo ""

# 5. Run tests (optional)
if [ "$RUN_TESTS" = true ]; then
    echo "[5/5] Running tests..."
    uv run python3 test_rag_pipeline.py
    echo "✓ Tests passed"
    echo ""
fi

# Start server
echo "=========================================="
echo "Starting Server (Local)"
echo "=========================================="
echo "URL: http://localhost:8000"
echo "Docs: http://localhost:8000/docs"
echo "Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "Endpoints:"
echo "  POST /answer   - Knowledge Assistant Endpoint"
echo "  POST /classify - Feedback classification endpoint"
echo "  GET  /health   - Health check"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Use uv to run the server
uv run python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
