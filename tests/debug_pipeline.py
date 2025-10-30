#!/usr/bin/env python3
"""
Debug script to trace why LLM ignores retrieved context
"""

from src.generation.query_analyzer import query_analyzer
from src.retrieval.langchain_retriever import langchain_retriever
from src.retrieval.jina_reranker import jina_reranker
from src.generation.context_builder import context_builder
from src.generation.answer_generator import answer_generator
from src.retrieval.vector_store import vector_store

# The problematic query
QUERY = "ما هي مدة معالجة طلب الاسترداد؟"
EXPECTED_CHUNK = "t0002"
EXPECTED_ANSWER = "5 إلى 10 أيام عمل"

print("=" * 100)
print("DEBUGGING: Why LLM ignores retrieved context")
print("=" * 100)
print(f"\nQuery: {QUERY}")
print(f"Expected chunk: {EXPECTED_CHUNK}")
print(f"Expected answer: {EXPECTED_ANSWER}")

# Initialize
print("\n[1] Initializing...")
chunks = vector_store.load_chunks_from_directory()
if not langchain_retriever.is_ready():
    langchain_retriever.initialize_bm25(chunks)
    langchain_retriever.initialize_vector_store()
print(f"✓ Loaded {len(chunks)} chunks")

# Step 1: Query Analysis
print("\n" + "=" * 100)
print("[2] QUERY ANALYSIS")
print("=" * 100)
analysis = query_analyzer.analyze(QUERY)
print(f"Intent: {analysis.intent}")
print(f"Language: {analysis.language}")
print(f"Doc Type: {analysis.doc_type}")
print(f"Filters: {analysis.filters}")

# Step 2: Retrieval
print("\n" + "=" * 100)
print("[3] HYBRID RETRIEVAL")
print("=" * 100)
retrieved = langchain_retriever.search(QUERY, filters=analysis.filters, top_k=20)
print(f"Retrieved {len(retrieved)} chunks")

# Check if t0002 is in retrieved chunks
found_t0002 = False
for i, chunk in enumerate(retrieved):
    chunk_id = chunk["payload"].get("chunk_id", "")
    if chunk_id == EXPECTED_CHUNK:
        found_t0002 = True
        print(f"\n✓ FOUND {EXPECTED_CHUNK} at position {i + 1}")
        print(f"  Content field: {chunk['payload'].get('content', 'N/A')[:100]}...")
        print(f"  Question field: {chunk['payload'].get('question', 'N/A')}")
        print(f"  Answer field: {chunk['payload'].get('answer', 'N/A')[:100]}...")
        print(f"  All payload keys: {list(chunk['payload'].keys())}")
        break

if not found_t0002:
    print(f"\n✗ {EXPECTED_CHUNK} NOT FOUND in retrieval results")
    print("\nTop 5 retrieved chunks:")
    for i, chunk in enumerate(retrieved[:5]):
        chunk_id = chunk["payload"].get("chunk_id", "no_id")
        content = chunk["payload"].get("content", "")[:80]
        print(f"  [{i + 1}] {chunk_id}: {content}...")

# Step 3: Reranking
print("\n" + "=" * 100)
print("[4] JINA RERANKING")
print("=" * 100)
reranked = jina_reranker.rerank(QUERY, retrieved, top_n=5)
print(f"Reranked to top {len(reranked)} chunks")

# Check if t0002 survived reranking
found_after_rerank = False
for i, chunk in enumerate(reranked):
    chunk_id = chunk["payload"].get("chunk_id", "")
    score = chunk.get("rerank_score", 0.0)
    if chunk_id == EXPECTED_CHUNK:
        found_after_rerank = True
        print(f"\n✓ {EXPECTED_CHUNK} SURVIVED RERANKING at position {i + 1}")
        print(f"  Rerank score: {score}")
        break

if not found_after_rerank:
    print(f"\n✗ {EXPECTED_CHUNK} LOST DURING RERANKING")
    print("\nTop 5 after reranking:")
    for i, chunk in enumerate(reranked[:5]):
        chunk_id = chunk["payload"].get("chunk_id", "no_id")
        score = chunk.get("rerank_score", 0.0)
        content = chunk["payload"].get("content", "")[:80]
        print(f"  [{i + 1}] Score: {score:.4f} | {chunk_id}: {content}...")

# Step 4: Context Building
print("\n" + "=" * 100)
print("[5] CONTEXT BUILDING")
print("=" * 100)
context_data = context_builder.build_context_with_metadata(reranked)

print(f"Chunks in context: {context_data['num_chunks']}")
print(f"Citations: {len(context_data['citations'])}")
print(f"Warnings: {len(context_data['warnings'])}")
print(f"Steps: {len(context_data['steps'])}")

print("\n--- FORMATTED CONTEXT SENT TO LLM ---")
print(context_data["formatted_context"])
print("--- END OF CONTEXT ---")

# Check if t0002 content appears in the formatted context
if EXPECTED_ANSWER in context_data["formatted_context"]:
    print(f"\n✓ Expected answer '{EXPECTED_ANSWER}' IS in the context!")
elif (
    "5" in context_data["formatted_context"]
    and "10" in context_data["formatted_context"]
):
    print("\n⚠ Numbers 5 and 10 are in context, but not the exact expected phrase")
else:
    print(f"\n✗ Expected answer '{EXPECTED_ANSWER}' NOT in context!")

# Step 5: Answer Generation
print("\n" + "=" * 100)
print("[6] ANSWER GENERATION")
print("=" * 100)

answer_result = answer_generator.generate(
    query=QUERY,
    context=context_data["formatted_context"],
    language="ar",
    citations_list=context_data["citations"],
    extracted_steps=context_data["steps"],
    extracted_warnings=context_data["warnings"],
    rerank_scores=[c.get("rerank_score", 0.0) for c in reranked],
)

print("\nGenerated answer:")
print(f"  {answer_result.answer}")
print(f"\nCitations: {len(answer_result.citations)}")
print(f"Confidence: {answer_result.confidence}")

# Final check
if EXPECTED_ANSWER in answer_result.answer or (
    "5" in answer_result.answer and "10" in answer_result.answer
):
    print("\n✓ ANSWER CONTAINS EXPECTED INFORMATION")
else:
    print("\n✗ ANSWER DOES NOT CONTAIN EXPECTED INFORMATION")

print("\n" + "=" * 100)
print("DEBUG COMPLETE")
print("=" * 100)
