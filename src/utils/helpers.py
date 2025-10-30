#!/usr/bin/env python3
"""
Helper utility functions
"""

import re
from typing import Any, Dict


def clean_text(text: str) -> str:
    """
    Clean and normalize text

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_chunk_content(chunk: Dict[str, Any]) -> str:
    """
    Extract content from a chunk dictionary

    Args:
        chunk: Chunk dictionary

    Returns:
        Content string
    """
    # Handle case where chunk might be a list (shouldn't happen, but defensive)
    if isinstance(chunk, list):
        if len(chunk) > 0 and isinstance(chunk[0], dict):
            chunk = chunk[0]  # Take first item
        else:
            return ""

    if not isinstance(chunk, dict):
        return ""

    content = chunk.get("content", "")

    # If it's a FAQ or ticket, combine question and answer
    if chunk.get("chunk_type") in ["faq", "resolved_ticket"]:
        question = chunk.get("question", "")
        answer = chunk.get("answer", "")
        if question and answer:
            content = f"Q: {question}\nA: {answer}"

    return clean_text(content)


def format_citation(doc_id: str, page_num: Any, section_title: str) -> str:
    """
    Format citation string

    Args:
        doc_id: Document ID
        page_num: Page number
        section_title: Section title

    Returns:
        Formatted citation string
    """
    return f"[{doc_id}:p{page_num}:{section_title}]"


def parse_citation(citation_str: str) -> Dict[str, Any]:
    """
    Parse citation string back to components

    Args:
        citation_str: Citation string in format [doc_id:pN:section]

    Returns:
        Dictionary with doc_id, page_num, section_title
    """
    pattern = r"\[([^:]+):p(\d+):([^\]]+)\]"
    match = re.search(pattern, citation_str)

    if match:
        return {
            "doc_id": match.group(1),
            "page_num": int(match.group(2)),
            "section_title": match.group(3),
        }

    return {}


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to maximum length

    Args:
        text: Input text
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."
