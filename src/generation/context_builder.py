#!/usr/bin/env python3
"""
Context assembly for RAG prompts
"""

from typing import Any, Dict, List

from src.utils.logger import logger


class ContextBuilder:
    """Build context for LLM prompts from retrieved chunks"""

    def __init__(self):
        """Initialize context builder"""
        logger.info("Initialized ContextBuilder")

    def format_chunk(self, chunk: Dict[str, Any], index: int) -> str:
        """
        Format a single chunk for context

        Args:
            chunk: Chunk data
            index: Chunk index (for reference)

        Returns:
            Formatted chunk string
        """
        payload = chunk.get("payload", {})

        doc_id = payload.get("doc_id", "unknown")
        page_num = payload.get("page_num", "")
        section_title = payload.get("section_title", "")
        chunk_type = payload.get("chunk_type", "text")

        # Get content based on chunk type
        # FAQ and resolved_ticket chunks have question/answer fields instead of content
        if chunk_type in ["faq", "resolved_ticket"]:
            question = payload.get("question", "")
            answer = payload.get("answer", "")
            content = (
                f"Q: {question}\nA: {answer}"
                if question and answer
                else payload.get("content", "")
            )
        else:
            content = payload.get("content", "")

        # Build header
        header_parts = [f"Document: {doc_id}"]
        if page_num:
            header_parts.append(f"Page: {page_num}")
        if section_title:
            header_parts.append(f"Section: {section_title}")

        header = "[" + ", ".join(header_parts) + "]"

        # Add chunk type indicator if not text
        if chunk_type != "text":
            header += f" (Type: {chunk_type})"

        # Format content
        formatted = f"{header}\n{content}\n"

        # Add visual information if available
        if chunk_type in ["flowchart", "chart", "visual"]:
            image_path = payload.get("image_path", "")
            if image_path:
                formatted += f"[Visual: {image_path}]\n"

        formatted += "---\n"

        return formatted

    def assemble_context(
        self, chunks: List[Dict[str, Any]], include_scores: bool = False
    ) -> str:
        """
        Assemble context from multiple chunks

        Args:
            chunks: List of retrieved chunks
            include_scores: Whether to include relevance scores

        Returns:
            Formatted context string
        """
        if not chunks:
            logger.warning("No chunks provided for context assembly")
            return ""

        logger.info(f"Assembling context from {len(chunks)} chunks")

        context_parts = []

        for idx, chunk in enumerate(chunks, start=1):
            formatted_chunk = self.format_chunk(chunk, idx)

            if include_scores:
                score = chunk.get(
                    "rerank_score", chunk.get("rrf_score", chunk.get("score"))
                )
                if score is not None:
                    formatted_chunk = f"[Relevance: {score:.3f}]\n" + formatted_chunk

            context_parts.append(formatted_chunk)

        context = "\n".join(context_parts)

        logger.debug(f"Context assembled: {len(context)} characters")
        return context

    def extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citation information from chunks

        Args:
            chunks: Retrieved chunks

        Returns:
            List of citation dictionaries
        """
        citations = []

        for chunk in chunks:
            payload = chunk.get("payload", {})

            citation = {
                "doc_id": payload.get("doc_id", ""),
                "page_num": payload.get("page_num"),
                "section_title": payload.get("section_title", ""),
                "chunk_id": payload.get("chunk_id", ""),
            }

            citations.append(citation)

        return citations

    def extract_warnings(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Extract warnings and hazards from chunks

        Args:
            chunks: Retrieved chunks

        Returns:
            List of warning strings
        """
        warnings = []
        warning_keywords = [
            "warning",
            "caution",
            "hazard",
            "danger",
            "alert",
            "important",
            "تحذير",
            "تنبيه",
            "خطر",
            "هام",  # Arabic warning keywords
        ]

        for chunk in chunks:
            payload = chunk.get("payload", {})
            chunk_type = payload.get("chunk_type", "text")

            # Get content based on chunk type
            if chunk_type in ["faq", "resolved_ticket"]:
                question = payload.get("question", "")
                answer = payload.get("answer", "")
                full_text = f"{question} {answer}"
            else:
                full_text = payload.get("content", "")

            content = full_text.lower()

            # Check if content contains warning keywords
            for keyword in warning_keywords:
                if keyword in content:
                    # Extract surrounding context (simplified - just get the text)
                    if full_text and full_text not in warnings:
                        warnings.append(full_text[:200])  # Limit length
                    break

        return warnings[:5]  # Limit to top 5 warnings

    def extract_steps(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Extract numbered steps from procedural content

        Args:
            chunks: Retrieved chunks

        Returns:
            List of step strings
        """
        steps = []

        for chunk in chunks:
            payload = chunk.get("payload", {})
            chunk_type = payload.get("chunk_type", "text")

            # Get content based on chunk type
            if chunk_type in ["faq", "resolved_ticket"]:
                question = payload.get("question", "")
                answer = payload.get("answer", "")
                content = f"{question}\n{answer}"
            else:
                content = payload.get("content", "")

            # Look for numbered steps in content
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                # Match patterns like "1.", "Step 1:", etc.
                if line and (
                    line[0].isdigit()
                    or line.lower().startswith("step")
                    or line.startswith("•")
                    or line.startswith("-")
                ):
                    if line not in steps:
                        steps.append(line)

        return steps

    def build_context_with_metadata(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build context with metadata for prompt

        Args:
            chunks: Retrieved chunks

        Returns:
            Dictionary with context and metadata
        """
        # Group chunks by type
        doc_chunks = []
        faq_chunks = []
        ticket_chunks = []
        flowchart_chunks = []

        for chunk in chunks:
            chunk_type = chunk["payload"].get("chunk_type", "")
            if chunk_type == "faq":
                faq_chunks.append(chunk)
            elif chunk_type == "resolved_ticket":
                ticket_chunks.append(chunk)
            elif chunk_type == "flowchart":
                flowchart_chunks.append(chunk)
            else:
                doc_chunks.append(chunk)

        # Build structured context
        context_parts = []

        if doc_chunks:
            context_parts.append("=== DOCUMENT CONTENT ===")
            context_parts.append(self.assemble_context(doc_chunks))

        if flowchart_chunks:
            context_parts.append("=== PROCESS FLOWCHARTS ===")
            context_parts.append(self.assemble_context(flowchart_chunks))

        if faq_chunks:
            context_parts.append("=== FAQ ===")
            context_parts.append(self.assemble_context(faq_chunks))

        if ticket_chunks:
            context_parts.append("=== RESOLVED TICKETS ===")
            context_parts.append(self.assemble_context(ticket_chunks))

        formatted_context = "\n\n".join(context_parts)

        return {
            "formatted_context": formatted_context,
            "citations": self.extract_citations(chunks),
            "warnings": self.extract_warnings(chunks),
            "steps": self.extract_steps(chunks),
            "num_chunks": len(chunks),
            "chunk_types": [c["payload"].get("chunk_type", "text") for c in chunks],
        }


# Singleton instance
context_builder = ContextBuilder()
