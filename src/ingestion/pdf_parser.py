#!/usr/bin/env python3
"""
PDF Extractor with Improved Image Extraction
Extracts text, images (with correct page attribution), and tables from PDFs for RAG ingestion
"""

import fitz  # PyMuPDF
import os
import json
import camelot
from pathlib import Path


def _has_valid_table_data(data):
    """
    Check if table data contains any non-empty content.

    Args:
        data: List of lists representing table rows

    Returns:
        bool: True if table has valid data
    """
    if not data:
        return False
    for row in data:
        for cell in row:
            if cell and str(cell).strip():  # Check if cell has non-empty content
                return True
    return False


def extract_all_from_pdf(file_path, output_dir="extracted_images", verbose=True):
    """
    Extract ALL content from a PDF: text, unique images, and tables

    Uses improved dict-based extraction for accurate image page attribution

    Args:
        file_path (str): Path to the PDF file
        output_dir (str): Directory to save extracted images
        verbose (bool): Print progress messages

    Returns:
        dict: Dictionary containing text, images info, and tables data
    """
    result = {
        "pdf_path": file_path,
        "pdf_name": os.path.basename(file_path),
        "text": [],
        "images": [],
        "tables": [],
        "num_pages": 0,
    }

    try:
        doc = fitz.open(file_path)
        result["num_pages"] = len(doc)

        # Get document name without extension (outside loops)
        doc_name = os.path.splitext(result["pdf_name"])[0]

        # Extract text from all pages
        if verbose:
            print(f"Extracting text from {result['num_pages']} pages...")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text", sort=True)
            result["text"].append({"page": page_num + 1, "text": text})

        if verbose:
            total_chars = sum(len(page["text"]) for page in result["text"])
            print(
                f"Extracted {total_chars} characters of text from {len(result['text'])} pages"
            )

        os.makedirs(output_dir, exist_ok=True)

        # Extract images using dict format for accurate page attribution
        if verbose:
            print("Extracting unique images...")

        image_count = 0
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_num_display = page_num + 1

            # Get blocks with dict format to access images on each page
            d = page.get_text("dict", sort=True)
            blocks = d["blocks"]

            for b in blocks:
                if b["type"] == 1:  # image block
                    try:
                        img_bytes = b["image"]
                        ext = b.get("ext", "png")
                        bbox = b["bbox"]  # (x0, y0, x1, y1)

                        # Correct filename with document name and actual page number
                        filename = f"{output_dir}/{doc_name}_page{page_num_display}_img{len(result['images']) + 1}.{ext}"

                        with open(filename, "wb") as imgf:
                            imgf.write(img_bytes)

                        result["images"].append(
                            {
                                "page": page_num_display,
                                "filename": filename,
                                "bbox": bbox,
                                "ext": ext,
                            }
                        )
                        image_count += 1
                    except Exception as e:
                        if verbose:
                            print(
                                f"Error extracting image from page {page_num_display}: {e}"
                            )

        if verbose:
            print(f"Extracted {image_count} unique image(s)")

        doc.close()

    except Exception as e:
        print(f"Error in PyMuPDF processing: {str(e)}")
        result["error"] = str(e)

    # Extract tables using Camelot
    if verbose:
        print("Extracting tables...")

    try:
        tables = camelot.read_pdf(file_path, pages="all", flavor="lattice")

        if tables:
            if verbose:
                print(f"Found {len(tables)} table(s)")

            for table_idx, table in enumerate(tables):
                page_num = table.page - 1  # Camelot uses 1-based page numbers
                df = table.df

                table_data = {
                    "table_idx": table_idx + 1,
                    "page": page_num + 1,
                    "shape": df.shape,
                    "data": df.values.tolist(),
                    "columns": df.columns.tolist(),
                }
                # Only add table if it has data (not empty or all empty strings)
                if _has_valid_table_data(table_data["data"]):
                    result["tables"].append(table_data)
        else:
            if verbose:
                print("No tables found")

    except Exception as e:
        if verbose:
            print(f"Error in table extraction: {str(e)}")
        result["table_extraction_error"] = str(e)

    return result


def process_all_pdfs(
    input_dir,
    output_dir="pdf_extracts",
    image_dir="extracted_images",
    metadata_file=None,
    verbose=True,
):
    """
    Process all PDF files in a directory

    Args:
        input_dir (str): Directory containing PDF files
        output_dir (str): Directory to save extracted data
        image_dir (str): Directory to save extracted images
        metadata_file (str): Path to doc_metadata.json file
        verbose (bool): Print progress messages

    Returns:
        list: List of processed PDF filenames
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return []

    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return []

    # Load metadata if provided
    metadata_lookup = {}
    if metadata_file and os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_list = json.load(f)
                # Create lookup dictionary by doc_id
                for meta in metadata_list:
                    metadata_lookup[meta["doc_id"]] = meta
                if verbose:
                    print(f"Loaded metadata for {len(metadata_lookup)} documents")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load metadata file: {e}")

    print(f"\nFound {len(pdf_files)} PDF file(s) to process\n")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    processed_files = []

    # Process each PDF
    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        print("-" * 80)

        # Extract content
        result = extract_all_from_pdf(
            str(pdf_file), output_dir=image_dir, verbose=verbose
        )

        # Add metadata if available
        doc_id = pdf_file.stem  # PDF name without extension
        if doc_id in metadata_lookup:
            result["doc_metadata"] = metadata_lookup[doc_id]
            if verbose:
                print(f"  Added metadata: {result['doc_metadata']}")

        # Save individual JSON file
        json_filename = os.path.join(output_dir, f"{pdf_file.stem}.json")
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if verbose:
            total_chars = sum(len(page["text"]) for page in result["text"])
            print(f"\n✓ Saved: {json_filename}")
            print(
                f"  Text: {total_chars} characters across {len(result['text'])} pages"
            )
            print(f"  Images: {len(result['images'])}")
            print(f"  Tables: {len(result['tables'])}")

        processed_files.append(pdf_file.name)

    print("\n" + "=" * 80)
    print("\n✓ Extraction complete!")
    print(f"✓ Individual JSON files saved in: {output_dir}/")
    print(f"✓ Images saved in: {image_dir}/")

    return processed_files


def extract_pdfs_for_ingestion(
    input_dir="data/docs",
    output_dir="data/pdf_extracts",
    image_dir="data/pdf_images",
    metadata_file="data/doc_metadata.json",
):
    """
    Simplified extraction function for ingestion pipeline.

    Args:
        input_dir: Directory with PDF files
        output_dir: Directory for JSON output
        image_dir: Directory for extracted images
        metadata_file: Path to doc_metadata.json

    Returns:
        list: Processed filenames
    """
    return process_all_pdfs(
        input_dir=input_dir,
        output_dir=output_dir,
        image_dir=image_dir,
        metadata_file=metadata_file,
        verbose=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract content from PDF files for RAG ingestion"
    )
    parser.add_argument("input_dir", help="Directory containing PDF files")
    parser.add_argument(
        "--output",
        "-o",
        default="pdf_extracts",
        help="Output directory for JSON files (default: pdf_extracts)",
    )
    parser.add_argument(
        "--images",
        "-i",
        default="extracted_images",
        help="Directory for extracted images (default: extracted_images)",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        default="data/doc_metadata.json",
        help="Path to doc_metadata.json file (default: data/doc_metadata.json)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress messages"
    )

    args = parser.parse_args()

    process_all_pdfs(
        args.input_dir,
        output_dir=args.output,
        image_dir=args.images,
        metadata_file=args.metadata,
        verbose=not args.quiet,
    )
