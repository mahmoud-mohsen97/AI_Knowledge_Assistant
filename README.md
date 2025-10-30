# AI Knowledge Assistant 2.0

RAG-based question answering system with LangChain + Jina AI reranker.

## Quick Start

```bash
# 1. Install/update dependencies (first time or after updates)
./fix_deps.sh

# 2. Restart terminal or activate venv
source .venv/bin/activate

# 3. Start the server
./start.sh
```

**First run:** Script will create `.env` - add your `OPENAI_API_KEY` and run again.

## Options

```bash
./start.sh --force-reindex  # Rebuild vector store
./start.sh --test           # Run tests
```

## API Endpoints

- `POST /answer` - Submit query (returns answer with citations, steps, warnings)
- `POST /classify` - Classify feedback
- `GET /health` - System health
- `GET /docs` - API documentation

## Example

```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is covered?", "language": "English"}'
```

## Response Format

```json
{
  "answer": "...",
  "citations": [{"doc_id": "...", "section": "...", "page_num": 2}],
  "steps": ["1. ...", "2. ..."],
  "warnings": [],
  "confidence": 0.95,
  "latency_ms": 1200,
  "token_cost_estimate": 450
}
```

## Features

- ✅ Enhanced query analysis (extracts metadata)
- ✅ Intelligent embedding (content + titles)
- ✅ LangChain hybrid retrieval (BM25 + Dense)
- ✅ Jina AI reranker (multilingual)
- ✅ Automatic step extraction (procedural queries)
- ✅ Warning detection (hazards/alerts)
- ✅ Performance metrics (latency, tokens, confidence)
- ✅ English + Arabic support

## Stack

- **Framework:** LangChain
- **Vector DB:** Qdrant
- **Embeddings:** OpenAI text-embedding-3-large
- **Reranker:** Jina AI (jina-reranker-v2-base-multilingual)
- **LLM:** gpt-4.1-mini
- **API:** FastAPI
