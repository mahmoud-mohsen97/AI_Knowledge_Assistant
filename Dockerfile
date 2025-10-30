FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files and source code
COPY pyproject.toml ./
COPY src/ ./src/
COPY main.py ./

# Install Python dependencies using uv with CPU-only PyTorch
# Install torch-cpu FIRST to avoid CUDA dependencies (saves 3.5GB)
RUN uv venv && \
    UV_INDEX_URL="https://download.pytorch.org/whl/cpu" uv pip install --no-cache torch && \
    uv pip install --no-cache -e .

# Copy data after install for better caching
COPY data/ ./data/

# Create models directory (models are downloaded at runtime or mounted)
RUN mkdir -p models

# Download NLTK data (use venv python directly to avoid re-downloading deps)
RUN .venv/bin/python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Expose API port
EXPOSE 8000

# Run the application using venv python directly (avoid uv run re-downloading deps)
CMD [".venv/bin/python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
