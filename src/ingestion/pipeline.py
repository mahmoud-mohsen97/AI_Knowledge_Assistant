#!/usr/bin/env python3
"""
Ingestion Pipeline
Sequential processing of documents: extraction, summarization, image interpretation and final chunking
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try imports with fallback
try:
    from src.ingestion.pdf_parser import extract_pdfs_for_ingestion
    from src.ingestion.summarizer import generate_summary, interpret_image
    from src.ingestion.chunker import (
        process_pdf_extract,
        process_faq_file,
        process_tickets_file,
        load_existing_visual_chunks,
    )
except ImportError:
    # Alternative import if running from within src/utils directory
    from ingestion.pdf_parser import extract_pdfs_for_ingestion
    from ingestion.summarizer import generate_summary, interpret_image
    from ingestion.chunker import (
        process_pdf_extract,
        process_faq_file,
        process_tickets_file,
        load_existing_visual_chunks,
    )


def process_document(
    json_file_path,
    image_dir="data/processed/pdf_images",
    output_dir="data/processed/visual_chunks",
    verbose=True,
):
    """
    Complete pipeline for processing a single document.

    Steps:
    1. Generate document summary
    2. Interpret associated images and save as visual chunks
    3. [Future: Add more steps here]

    Args:
        json_file_path: Path to JSON file
        image_dir: Directory containing images
        output_dir: Directory to save visual chunks
        verbose: Print progress

    Returns:
        dict: Processing results
    """

    result = {
        "file": os.path.basename(json_file_path),
        "summary": None,
        "images_processed": 0,
        "errors": [],
    }

    doc_name = os.path.splitext(os.path.basename(json_file_path))[0]
    json_dir = os.path.dirname(json_file_path)

    print(f"\n{'=' * 80}")
    print(f"Processing: {os.path.basename(json_file_path)}")
    print(f"{'=' * 80}")

    # Step 1: Generate Summary
    if verbose:
        print("\n[STEP 1] Generating document summary...")

    try:
        summary = generate_summary(json_file_path, verbose=verbose)
        if summary:
            result["summary"] = summary
    except Exception as e:
        result["errors"].append(f"Summary error: {e}")

    # Step 2: Interpret Images
    if verbose:
        print("\n[STEP 2] Interpreting images...")

    # Find images for this document
    image_files = []
    if os.path.exists(image_dir):
        for img_file in os.listdir(image_dir):
            if img_file.startswith(doc_name + "_page") and img_file.endswith(".png"):
                image_files.append(os.path.join(image_dir, img_file))

    if not image_files:
        if verbose:
            print("  No images found")
    else:
        if verbose:
            print(f"  Found {len(image_files)} image(s)")

        for image_path in image_files:
            try:
                interpret_image(
                    image_path=image_path,
                    json_dir=json_dir,
                    output_dir=output_dir,
                    verbose=verbose,
                )
                result["images_processed"] += 1
            except Exception as e:
                result["errors"].append(f"Image error: {e}")

    # Step 3: [Future steps can be added here]
    # if verbose:
    #     print("\n[STEP 3] ...")

    if verbose:
        print(f"\n{'=' * 80}")
        print(
            f"✓ Completed: {result['images_processed']} image(s), {len(result['errors'])} error(s)"
        )
        print(f"{'=' * 80}")

    return result


def create_all_chunks(
    json_dir="data/processed/pdf_extracts",
    visual_chunks_dir="data/processed/visual_chunks",
    faq_path="data/raw/faq.json",
    tickets_path="data/raw/tickets_resolved.txt",
    doc_metadata_path="data/raw/doc_metadata.json",
    all_chunks_dir="data/processed/all_chunks",
    verbose=True,
):
    """
    Create final chunks from all sources (PDF extracts, FAQ, tickets, visual chunks).

    Args:
        json_dir: Directory with PDF extract JSON files
        visual_chunks_dir: Directory with visual chunk JSON files
        faq_path: Path to FAQ JSON file
        tickets_path: Path to tickets TSV file
        doc_metadata_path: Path to document metadata JSON file
        all_chunks_dir: Directory to save all chunks
        verbose: Print progress

    Returns:
        int: Total number of chunks created
    """

    # Create output directory
    Path(all_chunks_dir).mkdir(parents=True, exist_ok=True)

    # Load document metadata
    metadata_map = {}
    if os.path.exists(doc_metadata_path):
        try:
            with open(doc_metadata_path, "r", encoding="utf-8") as f:
                doc_metadata_list = json.load(f)
            metadata_map = {item["doc_id"]: item for item in doc_metadata_list}
            if verbose:
                print(f"  Loaded metadata for {len(metadata_map)} documents")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load metadata: {e}")
    else:
        if verbose:
            print(f"  Warning: Metadata file not found: {doc_metadata_path}")

    all_chunks = []

    # 1. Process PDF extracts
    if verbose:
        print("  Processing PDF extracts...")
    pdf_extract_files = list(Path(json_dir).glob("*.json"))

    for pdf_file in pdf_extract_files:
        try:
            with open(pdf_file, "r", encoding="utf-8") as f:
                extract_data = json.load(f)

            # Get metadata for this document
            doc_id = extract_data.get("doc_metadata", {}).get("doc_id", "")

            # If doc_id is empty, derive it from the filename
            if not doc_id:
                doc_id = pdf_file.stem  # filename without extension

            metadata = metadata_map.get(doc_id, {})

            # If metadata is empty but we have a doc_id, populate basic metadata
            if not metadata and doc_id:
                metadata = {"doc_id": doc_id}

            chunks = process_pdf_extract(extract_data, metadata)
            all_chunks.extend(chunks)
            if verbose:
                print(f"    {pdf_file.name}: {len(chunks)} chunks")
        except Exception as e:
            if verbose:
                print(f"    Error processing {pdf_file.name}: {e}")

    # 2. Process FAQ file
    if verbose:
        print("  Processing FAQ file...")
    if os.path.exists(faq_path):
        try:
            with open(faq_path, "r", encoding="utf-8") as f:
                faq_data = json.load(f)

            faq_chunks = process_faq_file(faq_data)
            all_chunks.extend(faq_chunks)
            if verbose:
                print(f"    FAQ: {len(faq_chunks)} chunks")
        except Exception as e:
            if verbose:
                print(f"    Error processing FAQ: {e}")
    else:
        if verbose:
            print(f"    FAQ file not found: {faq_path}")

    # 3. Process tickets file
    if verbose:
        print("  Processing tickets file...")
    if os.path.exists(tickets_path):
        try:
            ticket_chunks = process_tickets_file(tickets_path)
            all_chunks.extend(ticket_chunks)
            if verbose:
                print(f"    Tickets: {len(ticket_chunks)} chunks")
        except Exception as e:
            if verbose:
                print(f"    Error processing tickets: {e}")
    else:
        if verbose:
            print(f"    Tickets file not found: {tickets_path}")

    # 4. Load existing visual chunks
    if verbose:
        print("  Loading visual chunks...")
    if os.path.exists(visual_chunks_dir):
        try:
            visual_chunks = load_existing_visual_chunks(visual_chunks_dir)
            all_chunks.extend(visual_chunks)
            if verbose:
                print(f"    Visual chunks: {len(visual_chunks)} chunks")
        except Exception as e:
            if verbose:
                print(f"    Error loading visual chunks: {e}")
    else:
        if verbose:
            print(f"    Visual chunks directory not found: {visual_chunks_dir}")

    # 5. Save all chunks
    if verbose:
        print(f"  Saving all chunks to {all_chunks_dir}...")

    # Option 1: Save as one large JSON file
    output_file = Path(all_chunks_dir) / "all_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"    Saved all chunks to {output_file}")

    # Option 2: Save individual chunk files
    for idx, chunk in enumerate(all_chunks):
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        chunk_file = Path(all_chunks_dir) / f"{chunk_id}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"    Saved {len(all_chunks)} individual chunk files")
        print(f"  Total chunks created: {len(all_chunks)}")

    return len(all_chunks)


def process_all_documents(
    json_dir="data/processed/pdf_extracts",
    image_dir="data/processed/pdf_images",
    output_dir="data/processed/visual_chunks",
    pdf_dir="data/raw/docs",
    extract_first=True,
    verbose=True,
    create_final_chunks=True,
    faq_path="data/raw/faq.json",
    tickets_path="data/raw/tickets_resolved.txt",
    doc_metadata_path="data/raw/doc_metadata.json",
    all_chunks_dir="data/processed/all_chunks",
):
    """
    Process all documents in a directory.

    Args:
        json_dir: Directory with JSON files
        image_dir: Directory with images
        output_dir: Directory to save visual chunks
        pdf_dir: Directory with PDF source files
        extract_first: Extract PDFs first if JSON files don't exist
        verbose: Print progress
        create_final_chunks: Create final chunks from all sources
        faq_path: Path to FAQ JSON file
        tickets_path: Path to tickets TSV file
        doc_metadata_path: Path to document metadata JSON file
        all_chunks_dir: Directory to save final chunks

    Returns:
        list: Results for each document
    """

    json_path = Path(json_dir)

    # Step 0: Extract PDFs if needed and enabled
    if extract_first:
        if verbose:
            print("\n[STEP 0] Extracting PDFs...")
        extract_pdfs_for_ingestion(
            input_dir=pdf_dir, output_dir=json_dir, image_dir=image_dir
        )

    if not json_path.exists():
        print(f"Error: Directory {json_dir} does not exist")
        return []

    json_files = list(json_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return []

    print(f"\n{'=' * 80}")
    print("INGESTION PIPELINE")
    print(f"{'=' * 80}")
    print(f"Documents: {len(json_files)}")
    print(f"Visual chunks output: {output_dir}")
    print(f"{'=' * 80}\n")

    results = []
    for idx, json_file in enumerate(json_files, 1):
        if verbose:
            print(f"\nDocument {idx}/{len(json_files)}")

        result = process_document(
            str(json_file), image_dir=image_dir, output_dir=output_dir, verbose=verbose
        )
        results.append(result)

    # Summary of document processing
    total_images = sum(r["images_processed"] for r in results)
    total_errors = sum(len(r["errors"]) for r in results)

    print(f"\n{'=' * 80}")
    print("DOCUMENT PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Documents: {len(results)}")
    print(f"Images processed: {total_images}")
    print(f"Errors: {total_errors}")
    print(f"Visual chunks saved to: {output_dir}")
    print(f"{'=' * 80}")

    # Step 3: Create final chunks
    if create_final_chunks:
        if verbose:
            print("\n[STEP 3] Creating final chunks...")

        try:
            create_all_chunks(
                json_dir=json_dir,
                visual_chunks_dir=output_dir,
                faq_path=faq_path,
                tickets_path=tickets_path,
                doc_metadata_path=doc_metadata_path,
                all_chunks_dir=all_chunks_dir,
                verbose=verbose,
            )
        except Exception as e:
            print(f"Error creating final chunks: {e}")
            if verbose:
                import traceback

                traceback.print_exc()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingestion pipeline: Extract PDFs, generate summaries, interpret images and create final chunks"
    )
    parser.add_argument(
        "--pdf-dir", "-p", default="data/raw/docs", help="PDF directory"
    )
    parser.add_argument(
        "--json-dir", "-j", default="data/processed/pdf_extracts", help="JSON directory"
    )
    parser.add_argument(
        "--image-dir", "-i", default="data/processed/pdf_images", help="Image directory"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/processed/visual_chunks",
        help="Visual chunks output directory",
    )
    parser.add_argument(
        "--all-chunks-dir",
        "-a",
        default="data/processed/all_chunks",
        help="All chunks output directory",
    )
    parser.add_argument(
        "--faq-path", "-f", default="data/raw/faq.json", help="FAQ JSON file path"
    )
    parser.add_argument(
        "--tickets-path",
        "-t",
        default="data/raw/tickets_resolved.txt",
        help="Tickets TSV file path",
    )
    parser.add_argument(
        "--metadata-path",
        "-m",
        default="data/raw/doc_metadata.json",
        help="Document metadata JSON file path",
    )
    parser.add_argument(
        "--no-extract", action="store_true", help="Skip PDF extraction step"
    )
    parser.add_argument(
        "--no-final-chunks", action="store_true", help="Skip final chunks creation step"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    process_all_documents(
        json_dir=args.json_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        pdf_dir=args.pdf_dir,
        extract_first=not args.no_extract,
        verbose=not args.quiet,
        create_final_chunks=not args.no_final_chunks,
        faq_path=args.faq_path,
        tickets_path=args.tickets_path,
        doc_metadata_path=args.metadata_path,
        all_chunks_dir=args.all_chunks_dir,
    )
