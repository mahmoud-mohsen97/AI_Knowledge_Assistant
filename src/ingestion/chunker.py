"""
Chunking utility for processing PDF extracts, FAQ, and resolved tickets.
Creates structured JSON chunks for downstream tasks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def process_pdf_extract(
    extract_data: Dict[str, Any], doc_metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Process a PDF extract file and create chunks for each page.

    Args:
        extract_data: The loaded JSON from pdf_extracts
        doc_metadata: Metadata from doc_metadata.json

    Returns:
        List of chunk dictionaries
    """
    chunks = []
    metadata = extract_data.get("doc_metadata", {})
    images = extract_data.get("images", [])
    tables = extract_data.get("tables", [])
    doc_summary = extract_data.get("doc_summary", "")

    # Get doc_id, language, region, type, version from metadata
    doc_id = metadata.get("doc_id", "")
    language = metadata.get("lang", "en")
    region = metadata.get("region", "UAE")
    doc_type = metadata.get("type", "policy")
    version = metadata.get("version", "1.0")

    # Track images and tables by page
    images_by_page = {img["page"]: img["filename"] for img in images}
    tables_by_page = {}
    for table in tables:
        page = table.get("page", 1)
        if page not in tables_by_page:
            tables_by_page[page] = []
        tables_by_page[page].append(table.get("data", []))

    pages = extract_data.get("text", [])

    doc_title = None
    page_num = 0

    for page_data in pages:
        page_num = page_data.get("page", 0)
        text = page_data.get("text", "").strip()

        if not text:
            continue

        lines = text.split("\n")
        lines = [line.strip() for line in lines if line.strip()]

        if page_num == 1:
            # First page: first line is doc_title, next line with text is section_title
            if lines:
                doc_title = lines[0]
                section_title = ""
                content_lines = []

                # Find first actual content line (skip doc_title)
                if len(lines) > 1:
                    section_title = lines[1]
                    content_lines = lines[2:]
                else:
                    content_lines = []

                chunk_text = "\n".join(content_lines)
            else:
                section_title = ""
                chunk_text = ""
        else:
            # Subsequent pages: first line is section_title
            if lines:
                section_title = lines[0]
                content_lines = lines[1:]
                chunk_text = "\n".join(content_lines)
            else:
                section_title = ""
                chunk_text = ""

        # Get image for this page if exists
        image_path = images_by_page.get(page_num, "")

        # Get tables for this page if exists
        page_tables = tables_by_page.get(page_num, [])

        # Normalize doc_type mapping
        normalized_doc_type = ""
        if doc_type in ["policy", "Policy"]:
            normalized_doc_type = "policy"
        elif doc_type in ["procedure", "Procedure"]:
            normalized_doc_type = "procedure"
        elif doc_type in ["process", "Process"]:
            normalized_doc_type = "process"
        else:
            normalized_doc_type = "policy"  # default

        chunk = {
            "chunk_id": f"{doc_id}_chunk_{page_num:02d}",
            "doc_id": doc_id,
            "doc_title": doc_title,
            "chunk_type": "doc_page",
            "doc_summary": doc_summary,
            "page_num": page_num,
            "section_title": section_title,
            "content": chunk_text,
            "language": language,
            "region": region,
            "doc_type": normalized_doc_type,
            "version": version,
            "image": image_path,
            "table": page_tables[0] if page_tables else [],
        }

        chunks.append(chunk)

    return chunks


def parse_source_doc(source_str: str) -> Dict[str, Any]:
    """
    Parse a source string like "PAY-Policy-UAE-2.2#Grace Period and Penalties"
    and extract doc_type and section_title.

    Args:
        source_str: Source string in format "DOC-Policy/DocType-REGION-VERSION#Section Title"

    Returns:
        Dictionary with parsed doc_type and section_title
    """
    # Default values
    doc_type = ""
    section_title = ""

    if not source_str:
        return {"doc_type": doc_type, "section_title": section_title}

    # Check if there's a # separator
    if "#" in source_str:
        parts = source_str.split("#", 1)
        section_title = parts[1] if len(parts) > 1 else ""

    # Extract doc_type from the source string
    # Look for "Policy", "Procedure", or "Process" in the string
    if "Policy" in source_str or "-Policy-" in source_str:
        doc_type = "policy"
    elif "Procedure" in source_str or "-Procedure-" in source_str:
        doc_type = "procedure"
    elif "Process" in source_str or "-Process-" in source_str:
        doc_type = "process"

    return {"doc_type": doc_type, "section_title": section_title}


def process_faq_file(faq_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process FAQ JSON file and create chunks for each Q-A pair.

    Args:
        faq_data: List of FAQ dictionaries

    Returns:
        List of chunk dictionaries
    """
    chunks = []

    for idx, faq_item in enumerate(faq_data):
        # Handle both formats: "question" and "q", "answer" and "a"
        question = faq_item.get("question") or faq_item.get("q", "")
        answer = faq_item.get("answer") or faq_item.get("a", "")
        lang = faq_item.get("lang", "en")

        # Get the doc field (original source)
        doc_source = faq_item.get("doc", "")

        # Parse the doc field to extract doc_type and section_title
        parsed = parse_source_doc(doc_source)

        chunk = {
            "chunk_id": f"faq_{idx:03d}",
            "doc_id": "faq_database",
            "chunk_type": "faq",
            "question": question,
            "answer": answer,
            "source": doc_source,
            "doc_type": parsed["doc_type"] if parsed["doc_type"] else "faq",
            "section_title": parsed["section_title"],
            "language": lang,
        }

        chunks.append(chunk)

    return chunks


def process_tickets_file(tickets_path: str) -> List[Dict[str, Any]]:
    """
    Process resolved tickets TSV file and create chunks for each ticket.

    Args:
        tickets_path: Path to tickets_resolved.txt

    Returns:
        List of chunk dictionaries
    """
    chunks = []

    with open(tickets_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header line
    for idx, line in enumerate(lines[1:], start=1):
        line = line.strip()
        if not line:
            continue

        # More robust parsing: the format is id, lang, q, a, source
        # We need to split carefully since some fields may contain tabs

        # The last two tabs are always: answer, source
        # Before that: id (first tab), lang (second tab), question (everything else)

        # Find all tab positions
        tabs = [i for i, c in enumerate(line) if c == "\t"]

        if len(tabs) < 4:
            continue

        # First tab: ticket_id
        ticket_id = line[: tabs[0]]

        # Second tab: lang
        lang = line[tabs[0] + 1 : tabs[1]]

        # Last two tabs before end: answer and source
        # Everything between second tab and third-to-last tab is the question
        # Between third-to-last and second-to-last is the answer
        # After second-to-last is the source

        question = line[tabs[1] + 1 : tabs[-2]]
        answer = line[tabs[-2] + 1 : tabs[-1]]
        source = line[tabs[-1] + 1 :]

        # Parse the source field to extract doc_type and section_title
        parsed = parse_source_doc(source)

        chunk = {
            "chunk_id": ticket_id,
            "doc_id": "historical_tickets",
            "chunk_type": "resolved_ticket",
            "question": question,
            "answer": answer,
            "source": source,
            "doc_type": parsed["doc_type"] if parsed["doc_type"] else "ticket",
            "section_title": parsed["section_title"],
            "language": lang,
        }

        chunks.append(chunk)

    return chunks


def load_existing_visual_chunks(visual_chunks_dir: str) -> List[Dict[str, Any]]:
    """
    Load existing visual chunks from the visual_chunks directory.

    Args:
        visual_chunks_dir: Path to visual_chunks directory

    Returns:
        List of visual chunk dictionaries
    """
    chunks = []
    chunk_files = list(Path(visual_chunks_dir).glob("*.json"))

    for chunk_file in chunk_files:
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
                chunks.append(chunk_data)
        except Exception as e:
            print(f"Error loading visual chunk {chunk_file}: {e}")

    return chunks


def main():
    """Main function to process all files and generate chunks."""

    # Paths
    data_dir = Path("data")
    pdf_extracts_dir = data_dir / "pdf_extracts"
    faq_path = data_dir / "faq.json"
    tickets_path = data_dir / "tickets_resolved.txt"
    visual_chunks_dir = data_dir / "visual_chunks"
    output_dir = data_dir / "all_chunks"
    doc_metadata_path = data_dir / "doc_metadata.json"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Load document metadata
    with open(doc_metadata_path, "r", encoding="utf-8") as f:
        doc_metadata_list = json.load(f)

    # Create metadata lookup
    metadata_map = {item["doc_id"]: item for item in doc_metadata_list}

    all_chunks = []

    # 1. Process PDF extracts
    print("Processing PDF extracts...")
    pdf_extract_files = list(pdf_extracts_dir.glob("*.json"))

    for pdf_file in pdf_extract_files:
        try:
            with open(pdf_file, "r", encoding="utf-8") as f:
                extract_data = json.load(f)

            # Get metadata for this document
            doc_id = extract_data.get("doc_metadata", {}).get("doc_id", "")
            metadata = metadata_map.get(doc_id, {})

            chunks = process_pdf_extract(extract_data, metadata)
            all_chunks.extend(chunks)
            print(f"  Processed {pdf_file.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {e}")

    # 2. Process FAQ file
    print("\nProcessing FAQ file...")
    try:
        with open(faq_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)

        faq_chunks = process_faq_file(faq_data)
        all_chunks.extend(faq_chunks)
        print(f"  Processed FAQ: {len(faq_chunks)} chunks")
    except Exception as e:
        print(f"  Error processing FAQ: {e}")

    # 3. Process tickets file
    print("\nProcessing tickets file...")
    try:
        ticket_chunks = process_tickets_file(str(tickets_path))
        all_chunks.extend(ticket_chunks)
        print(f"  Processed tickets: {len(ticket_chunks)} chunks")
    except Exception as e:
        print(f"  Error processing tickets: {e}")

    # 4. Load existing visual chunks
    print("\nLoading visual chunks...")
    try:
        visual_chunks = load_existing_visual_chunks(str(visual_chunks_dir))

        # Add doc_summary to visual chunks by loading the corresponding PDF extract
        pdf_extract_files_map = {}
        for pdf_file in pdf_extracts_dir.glob("*.json"):
            try:
                with open(pdf_file, "r", encoding="utf-8") as f:
                    extract_data = json.load(f)
                doc_id = extract_data.get("doc_metadata", {}).get("doc_id", "")
                if doc_id:
                    pdf_extract_files_map[doc_id] = extract_data.get("doc_summary", "")
            except Exception:
                pass

        # Add doc_summary to visual chunks
        for visual_chunk in visual_chunks:
            v_doc_id = visual_chunk.get("doc_id", "")
            if v_doc_id in pdf_extract_files_map:
                visual_chunk["doc_summary"] = pdf_extract_files_map[v_doc_id]

        all_chunks.extend(visual_chunks)
        print(f"  Loaded visual chunks: {len(visual_chunks)} chunks")
    except Exception as e:
        print(f"  Error loading visual chunks: {e}")

    # 5. Save all chunks
    print(f"\nSaving all chunks to {output_dir}...")

    # Option 1: Save as one large JSON file
    output_file = output_dir / "all_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"  Saved all chunks to {output_file}")

    # Option 2: Save individual chunk files
    for idx, chunk in enumerate(all_chunks):
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        chunk_file = output_dir / f"{chunk_id}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(all_chunks)} individual chunk files")
    print(f"\nTotal chunks created: {len(all_chunks)}")


if __name__ == "__main__":
    main()
