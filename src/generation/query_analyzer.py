#!/usr/bin/env python3
"""
LLM-based query analysis for intent detection and metadata extraction
"""

import json
from typing import Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from src.config.settings import config
from src.config.prompts import (
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    get_query_analysis_user_prompt,
)
from src.utils.logger import logger


class QueryAnalysis(BaseModel):
    """Query analysis result schema"""

    intent: str = Field(
        description="Query intent: procedural, policy_lookup, process_flow, factual, visual_interpretation, numerical_query"
    )
    language: str = Field(description="Query language code (e.g., en, ar)")
    doc_type: Optional[str] = Field(
        default=None, description="Document type: policy, process, procedure"
    )
    section_title: Optional[str] = Field(
        default=None, description="Specific section mentioned in query"
    )
    chunk_type: Optional[str] = Field(
        default=None,
        description="Chunk type: doc_page, flowchart, chart, faq, resolved_ticket",
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Suggested metadata filters"
    )
    is_procedural: bool = Field(
        default=False, description="Whether query requires step-by-step instructions"
    )
    needs_warnings: bool = Field(
        default=False, description="Whether query might involve hazards/warnings"
    )
    confidence: str = Field(
        default="medium", description="Confidence level: high, medium, low"
    )


class QueryAnalyzer:
    """Analyze queries using LLM to detect intent and extract metadata"""

    def __init__(self, model: str = None):
        """
        Initialize query analyzer

        Args:
            model: LLM model name
        """
        self.model = model or config.LLM_MODEL
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        logger.info(f"Initialized QueryAnalyzer with model: {self.model}")

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query to extract intent, language, and metadata

        Args:
            query: User query

        Returns:
            QueryAnalysis object
        """
        logger.info(f"Analyzing query: '{query[:100]}...'")

        # Use centralized prompts
        system_prompt = QUERY_ANALYSIS_SYSTEM_PROMPT
        user_prompt = get_query_analysis_user_prompt(query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            result_dict = json.loads(result_text)

            analysis = QueryAnalysis(**result_dict)
            logger.info(
                f"Query analysis: intent={analysis.intent}, language={analysis.language}"
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return default analysis
            return QueryAnalysis(
                intent="factual",
                language="en",
                doc_type=None,
                section_title=None,
                chunk_type=None,
                filters={},
                is_procedural=False,
                needs_warnings=False,
                confidence="low",
            )


# Singleton instance
query_analyzer = QueryAnalyzer()
