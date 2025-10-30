#!/usr/bin/env python3
"""
Answer generation using LLM with citations
"""

import json
from typing import Dict, Any, List
from openai import OpenAI
from pydantic import BaseModel, Field
from src.config.settings import config
from src.config.prompts import (
    get_answer_generation_system_prompt,
    get_answer_generation_user_prompt,
    FEW_SHOT_EXAMPLES,
)
from src.utils.logger import logger


class Answer(BaseModel):
    """Answer schema with citations, steps, warnings, and metrics"""

    answer: str = Field(description="Generated answer with inline citations")
    citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of citation details"
    )
    steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step instructions for procedural queries",
    )
    warnings: List[str] = Field(
        default_factory=list, description="Important warnings or hazards"
    )
    confidence: float = Field(default=0.5, description="Confidence score (0-1)")
    latency_ms: int = Field(default=0, description="Processing time in milliseconds")
    token_cost_estimate: int = Field(default=0, description="Estimated token cost")


class AnswerGenerator:
    """Generate answers using LLM with proper citations"""

    def __init__(self, model: str = None):
        """
        Initialize answer generator

        Args:
            model: LLM model name
        """
        self.model = model or config.LLM_MODEL
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        logger.info(f"Initialized AnswerGenerator with model: {self.model}")

    def _build_system_prompt(self, language: str = "en") -> str:
        """
        Build system prompt for answer generation

        Args:
            language: Response language

        Returns:
            System prompt string
        """
        # Use centralized prompt function
        return get_answer_generation_system_prompt(language)

    def _build_user_prompt(self, query: str, context: str) -> str:
        """
        Build user prompt with query and context

        Args:
            query: User query
            context: Formatted context from retrieved chunks

        Returns:
            User prompt string
        """
        # Use centralized prompt function
        return get_answer_generation_user_prompt(query, context)

    def generate(
        self,
        query: str,
        context: str,
        language: str = "en",
        citations_list: List[Dict[str, Any]] = None,
        extracted_steps: List[str] = None,
        extracted_warnings: List[str] = None,
        rerank_scores: List[float] = None,
    ) -> Answer:
        """
        Generate answer with citations, steps, warnings, and metrics

        Args:
            query: User query
            context: Formatted context string
            language: Response language
            citations_list: Pre-extracted citations from chunks
            extracted_steps: Pre-extracted steps from context
            extracted_warnings: Pre-extracted warnings from context
            rerank_scores: Reranking scores for confidence calculation

        Returns:
            Answer object with text, citations, steps, warnings, and metrics
        """
        import time

        start_time = time.time()

        logger.info(f"Generating answer for query: '{query[:100]}...'")

        system_prompt = self._build_system_prompt(language)
        user_prompt = self._build_user_prompt(query, context)

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

            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)

            # Estimate token cost
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = (
                response.usage.completion_tokens if response.usage else 0
            )
            token_cost_estimate = prompt_tokens + completion_tokens

            # Calculate confidence score from string or use rerank scores
            confidence_str = result_dict.get("confidence", "medium")
            if isinstance(confidence_str, str):
                confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
                confidence = confidence_map.get(confidence_str.lower(), 0.5)
            else:
                confidence = float(confidence_str)

            # Boost confidence if rerank scores are high
            if rerank_scores and len(rerank_scores) > 0:
                avg_rerank_score = sum(rerank_scores[:3]) / min(len(rerank_scores), 3)
                confidence = min(0.99, confidence * 0.7 + avg_rerank_score * 0.3)

            # Create Answer object
            answer = Answer(
                answer=result_dict.get("answer", ""),
                citations=result_dict.get("citations", []),
                steps=extracted_steps or [],
                warnings=extracted_warnings or [],
                confidence=round(confidence, 2),
                latency_ms=latency_ms,
                token_cost_estimate=token_cost_estimate,
            )

            # Merge with pre-extracted citations if provided
            if citations_list:
                existing_ids = {
                    c.get("chunk_id") for c in answer.citations if c.get("chunk_id")
                }
                for citation in citations_list:
                    if citation.get("chunk_id") not in existing_ids:
                        answer.citations.append(citation)

            logger.info(
                f"Answer generated: {len(answer.citations)} citations, {len(answer.steps)} steps, {len(answer.warnings)} warnings"
            )
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Return fallback answer
            latency_ms = int((time.time() - start_time) * 1000)
            return Answer(
                answer="I apologize, but I encountered an error while generating the answer. Please try again.",
                citations=[],
                steps=[],
                warnings=[],
                confidence=0.0,
                latency_ms=latency_ms,
                token_cost_estimate=0,
            )

    def generate_with_few_shot(
        self, query: str, context: str, language: str = "en"
    ) -> Answer:
        """
        Generate answer with few-shot examples

        Args:
            query: User query
            context: Formatted context
            language: Response language

        Returns:
            Answer object
        """
        system_prompt = self._build_system_prompt(language)

        # Use few-shot examples from centralized prompts
        few_shot_examples = FEW_SHOT_EXAMPLES.copy()

        user_prompt = self._build_user_prompt(query, context)

        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(few_shot_examples)
            messages.append({"role": "user", "content": user_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            result_dict = json.loads(result_text)

            answer = Answer(**result_dict)
            logger.info(
                f"Answer generated (few-shot) with {len(answer.citations)} citations"
            )

            return answer

        except Exception as e:
            logger.error(f"Error generating answer with few-shot: {e}")
            return self.generate(query, context, language)


# Singleton instance
answer_generator = AnswerGenerator()
