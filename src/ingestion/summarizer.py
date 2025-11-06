#!/usr/bin/env python3
"""
Document Summarizer and Image Interpreter
Core functions for document processing
"""

import json
import os
import re
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from src.config.prompts import (
    DOCUMENT_SUMMARY_SYSTEM_PROMPT,
    get_document_summary_user_prompt,
    get_image_interpretation_prompt,
)

load_dotenv()


def generate_summary(json_file_path, model="gpt-5-nano", verbose=True):
    """Generate a summary for a document JSON file."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return None

    # Load and extract text
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    full_text = "\n\n".join(page.get("text", "") for page in data.get("text", []))
    if not full_text:
        return None

    doc_metadata = data.get("doc_metadata", {})
    metadata_str = ", ".join([f"{k}: {v}" for k, v in doc_metadata.items() if v])

    # Get prompts from centralized location
    system_prompt = DOCUMENT_SUMMARY_SYSTEM_PROMPT
    user_prompt = get_document_summary_user_prompt(metadata_str, full_text)

    # Generate summary
    if verbose:
        print("Calling OpenAI API...")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    summary = response.choices[0].message.content.strip()

    # Save summary to JSON
    data["doc_summary"] = summary
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"✓ Summary generated ({len(summary)} characters)")

    return summary


def interpret_image(
    image_path,
    json_dir="data/pdf_extracts",
    output_dir="data/visual_chunks",
    verbose=True,
):
    """Interpret an image and save as a separate visual chunk JSON file."""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return None

    # Parse filename: doc_name_page3_img1.png -> doc_name, page 3
    filename = os.path.basename(image_path)
    match = re.match(r"^(.+?)_page(\d+)_img(\d+)\.", filename)
    if not match:
        print(f"Error: Cannot parse filename: {filename}")
        return None

    doc_name = match.group(1)
    page_num = int(match.group(2))
    img_num = int(match.group(3))

    # Load corresponding JSON
    json_file_path = os.path.join(json_dir, f"{doc_name}.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract context
    doc_summary = data.get("doc_summary", "")
    page_text = ""
    for page_data in data.get("text", []):
        if page_data.get("page") == page_num:
            page_text = page_data.get("text", "")
            break

    # Extract section_title from first line of page text
    section_title = doc_name  # Default fallback
    if page_text:
        first_line = page_text.split("\n")[0].strip()
        if first_line:
            section_title = first_line[:100]  # Limit to 100 chars

    # Extract doc_type from filename
    doc_type = "Document"  # Default fallback
    if "Procedure" in doc_name:
        doc_type = "Procedure"
    elif "Process" in doc_name:
        doc_type = "Process"
    elif "Policy" in doc_name:
        doc_type = "Policy"

    # Get doc_id from metadata
    doc_id = data.get("doc_metadata", {}).get("doc_id", doc_name)
    language = data.get("doc_metadata", {}).get("lang", "en")

    if verbose:
        print(f"Interpreting {filename} for {doc_name}, page {page_num}")

    # Get prompt from centralized location
    prompt = get_image_interpretation_prompt(doc_summary, page_text)

    # Call Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    with open(image_path, "rb") as f:
        image_data = f.read()

    response = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": image_data}]
    )

    interpretation = response.text.strip()

    # Determine visual type from interpretation
    chunk_type = "visual"
    if "flowchart" in interpretation.lower():
        chunk_type = "flowchart"
    elif "chart" in interpretation.lower() or "graph" in interpretation.lower():
        chunk_type = "chart"
    elif "diagram" in interpretation.lower():
        chunk_type = "diagram"
    elif "table" in interpretation.lower():
        chunk_type = "table"

    # Create visual chunk schema
    chunk_id = f"{doc_id}_visual_{page_num:02d}_{img_num:02d}"

    visual_chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "page_num": page_num,
        "section_title": section_title,
        "doc_summary": doc_summary,
        "doc_type": doc_type,
        "chunk_type": chunk_type,
        "content": interpretation,
        "image_path": filename,
        "language": language,
    }

    # Save to separate file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{chunk_id}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(visual_chunk, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"✓ Saved visual chunk: {output_file}")

    return visual_chunk
