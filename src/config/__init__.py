"""Configuration module"""

from src.config.settings import config, Config
from src.config.prompts import (
    # Query Analysis
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    QUERY_ANALYSIS_USER_PROMPT_TEMPLATE,
    get_query_analysis_user_prompt,
    # Answer Generation
    get_answer_generation_system_prompt,
    get_answer_generation_user_prompt,
    ANSWER_GENERATION_USER_PROMPT_TEMPLATE,
    FEW_SHOT_EXAMPLES,
    # Document Processing
    DOCUMENT_SUMMARY_SYSTEM_PROMPT,
    get_document_summary_user_prompt,
    get_image_interpretation_prompt,
)

__all__ = [
    "config",
    "Config",
    # Query Analysis
    "QUERY_ANALYSIS_SYSTEM_PROMPT",
    "QUERY_ANALYSIS_USER_PROMPT_TEMPLATE",
    "get_query_analysis_user_prompt",
    # Answer Generation
    "get_answer_generation_system_prompt",
    "get_answer_generation_user_prompt",
    "ANSWER_GENERATION_USER_PROMPT_TEMPLATE",
    "FEW_SHOT_EXAMPLES",
    # Document Processing
    "DOCUMENT_SUMMARY_SYSTEM_PROMPT",
    "get_document_summary_user_prompt",
    "get_image_interpretation_prompt",
]
