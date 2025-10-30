#!/usr/bin/env python3
"""
Generate answer predictions from eval_seed.txt by calling the API
"""

import csv
import time
import requests
from typing import Dict, List, Optional
import sys


API_URL = "http://localhost:8000/answer"
EVAL_SEED_PATH = "data/raw/eval_seed.txt"
OUTPUT_PATH = "data/predictions/answer_predictions.csv"

# Language mapping
LANG_MAP = {
    "en": "English",
    "ar": "Arabic",
}


def read_eval_seed(file_path: str) -> List[Dict[str, str]]:
    """Read eval_seed.txt and return list of records"""
    records = []
    line_number = 0

    with open(file_path, "r", encoding="utf-8") as f:
        # Skip header
        next(f)

        for line in f:
            line = line.strip()
            if not line:
                continue

            line_number += 1
            parts = line.split("\t")

            # Handle cases where some fields might be missing
            if len(parts) >= 2:
                # Check if first column is an ID or language code
                if parts[0].strip() and parts[1].strip():
                    # If we have at least 2 parts, figure out the structure
                    if len(parts) >= 3 and parts[0].strip() not in ["en", "ar"]:
                        # Format: id, lang, query
                        record_id = parts[0].strip()
                        lang = parts[1].strip()
                        query = parts[2].strip()
                    else:
                        # Format: lang, query (no id)
                        record_id = f"auto_{line_number:03d}"
                        lang = parts[0].strip()
                        query = parts[1].strip()

                    # Skip if query is empty
                    if query:
                        records.append({"id": record_id, "lang": lang, "query": query})

    return records


def call_api(query: str, language: str) -> Optional[Dict]:
    """Call the /answer API endpoint"""
    try:
        response = requests.post(
            API_URL,
            json={"query": query, "language": language, "skip_query_analysis": False},
            timeout=60,
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print(
            "ERROR: Cannot connect to API. Make sure the server is running at http://localhost:8000"
        )
        print("Run: ./start.sh")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"Timeout for query: {query[:50]}...")
        return None
    except Exception as e:
        print(f"Error calling API: {e}")
        return None


def format_citations(citations: List[Dict]) -> str:
    """Format citations as 'doc_id#section' separated by semicolons"""
    formatted = []

    for citation in citations:
        doc_id = citation.get("doc_id", "")
        section = citation.get("section", "")

        if doc_id and section:
            formatted.append(f"{doc_id}#{section}")
        elif doc_id:
            formatted.append(doc_id)

    return "; ".join(formatted)


def generate_predictions():
    """Main function to generate predictions"""
    print("=" * 80)
    print("Generating Answer Predictions")
    print("=" * 80)

    # Read eval seed
    print(f"\n📖 Reading {EVAL_SEED_PATH}...")
    records = read_eval_seed(EVAL_SEED_PATH)
    print(f"Found {len(records)} queries to process")

    # Process each query
    results = []
    successful = 0
    failed = 0

    print(f"\n🚀 Processing queries...")
    print("-" * 80)

    for i, record in enumerate(records, 1):
        record_id = record["id"]
        lang = record["lang"]
        query = record["query"]

        # Map language code
        language = LANG_MAP.get(lang, "English")

        print(f"\n[{i}/{len(records)}] ID: {record_id} | Lang: {lang}")
        print(f"Query: {query[:80]}{'...' if len(query) > 80 else ''}")

        # Call API
        response = call_api(query, language)

        if response:
            answer = response.get("answer", "")
            citations = response.get("citations", [])
            confidence = response.get("confidence", 0.0)
            latency_ms = response.get("latency_ms", 0)

            formatted_citations = format_citations(citations)

            results.append(
                {"id": record_id, "answer": answer, "citations": formatted_citations}
            )

            print(
                f"✅ Success | Confidence: {confidence:.2f} | Latency: {latency_ms}ms"
            )
            print(
                f"Citations: {formatted_citations if formatted_citations else 'None'}"
            )
            successful += 1
        else:
            # Store empty result for failed queries
            results.append({"id": record_id, "answer": "", "citations": ""})
            print(f"❌ Failed")
            failed += 1

        # Small delay to avoid overwhelming the API
        time.sleep(0.5)

    # Write results to CSV
    print(f"\n💾 Writing results to {OUTPUT_PATH}...")

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer", "citations"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print("\n" + "=" * 80)
    print("✨ DONE!")
    print("=" * 80)
    print(f"Total queries: {len(records)}")
    print(f"Successful: {successful} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Output file: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    generate_predictions()
