#!/usr/bin/env python3
"""
Pydantic schemas for API request/response models
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Query request schema"""

    query: str = Field(..., description="User query", min_length=1)
    language: str = Field(
        default="English", description="Response language: 'English' or 'Arabic'"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata filters"
    )
    skip_query_analysis: bool = Field(
        default=False, description="Skip LLM-based query analysis"
    )
    use_few_shot: bool = Field(default=False, description="Use few-shot prompting")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the deadline for submitting an appeal?",
                "language": "English",
                "filters": {"region": "KSA", "doc_type": "process"},
            }
        }


class Citation(BaseModel):
    """Citation schema"""

    doc_id: str
    section: str  # Renamed from section_title
    page_num: Optional[int] = None
    chunk_id: Optional[str] = None


class AnswerResponse(BaseModel):
    """Answer response schema"""

    answer: str = Field(description="Generated answer with inline citations")
    citations: List[Citation] = Field(description="List of source citations")
    steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step instructions for procedural queries",
    )
    warnings: List[str] = Field(
        default_factory=list, description="Important warnings or hazards identified"
    )
    confidence: float = Field(description="Confidence score (0-1)")
    latency_ms: int = Field(description="Processing time in milliseconds")
    token_cost_estimate: int = Field(description="Estimated token cost")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The appeal must be submitted within 30 days from the decision date [APP-Process-KSA-2.1:Appeal Flow].",
                "citations": [
                    {
                        "doc_id": "APP-Process-KSA-2.1",
                        "section": "Appeal Flow",
                        "page_num": 2,
                        "chunk_id": "APP-Process-KSA-2.1_chunk_02",
                    }
                ],
                "steps": [
                    "1. Gather required documents",
                    "2. Submit appeal form within 30 days",
                    "3. Wait for review",
                ],
                "warnings": [],
                "confidence": 0.95,
                "latency_ms": 1250,
                "token_cost_estimate": 450,
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema"""

    status: str
    components: Dict[str, str]
    vector_store_count: int


class StatsResponse(BaseModel):
    """Statistics response schema"""

    total_chunks: int
    collection_name: str
    qdrant_host: str
    qdrant_port: int


class IndexRequest(BaseModel):
    """Index request schema"""

    force_reindex: bool = Field(
        default=False, description="Force reindexing even if collection exists"
    )


class IndexResponse(BaseModel):
    """Index response schema"""

    status: str
    message: str
    chunks_indexed: int


class ClassifyRequest(BaseModel):
    """Feedback classification request schema"""

    text: str = Field(..., description="Feedback text to classify", min_length=1)
    language: str = Field(default="en", description="Language code: 'en' or 'ar'")

    class Config:
        json_schema_extra = {
            "example": {"text": "I can't log in! OTP doesn't arrive!", "language": "en"}
        }


class ClassifyResponse(BaseModel):
    """Feedback classification response schema"""

    text: str = Field(description="Original feedback text")
    language: str = Field(description="Language code")
    level1: str = Field(description="Level 1 category (Technical, Payment, Claims)")
    level1_confidence: float = Field(description="Confidence score for level 1 (0-1)")
    level2: str = Field(description="Level 2 subcategory")
    level2_confidence: float = Field(description="Confidence score for level 2 (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I can't log in! OTP doesn't arrive!",
                "language": "en",
                "level1": "Technical",
                "level1_confidence": 0.9672,
                "level2": "Login",
                "level2_confidence": 0.8687,
            }
        }
