#!/usr/bin/env python3
"""
Test RAG Pipeline Components - Simple Sequential Testing
Shows output of each component step-by-step for assessment
"""

import asyncio

from src.generation.query_analyzer import query_analyzer
from src.retrieval.embedder import embedding_generator
from src.retrieval.langchain_retriever import langchain_retriever
from src.retrieval.jina_reranker import jina_reranker
from src.generation.context_builder import context_builder
from src.generation.rag_pipeline_new import new_rag_pipeline
from src.retrieval.vector_store import vector_store


# Test Questions (Fixed set from FAQ and Resolved Tickets)
TEST_QUESTIONS = [
    {
        "id": "faq_en_1",
        "source": "FAQ",
        "language": "English",
        "question": "What is the grace period for premium payments?",
        "expected_answer": "30 days from the due date",
    },
    {
        "id": "ticket_ar_1",
        "source": "Resolved Ticket",
        "language": "Arabic",
        "question": "ما هي مدة معالجة طلب الاسترداد؟",
        "expected_answer": "5-10 أيام عمل",
    },
    {
        "id": "ticket_en_1",
        "source": "Resolved Ticket",
        "language": "English",
        "question": "How do I file an appeal?",
        "expected_answer": "File within 30 days with explanation and evidence",
    },
    {
        "id": "faq_ar_1",
        "source": "FAQ",
        "language": "Arabic",
        "question": "هل يغطي التأمين الصحي الأمراض المزمنة؟",
        "expected_answer": "نعم، مثل السكري وارتفاع ضغط الدم",
    },
    {
        "id": "ticket_en_2",
        "source": "Resolved Ticket",
        "language": "English",
        "question": "What are the exclusions under the motor insurance policy?",
        "expected_answer": "Illegal racing, driving without licence, deliberate acts",
    },
]


def print_section_header(title: str, level: int = 1):
    """Print formatted section header"""
    if level == 1:
        print("\n" + "=" * 100)
        print(f"  {title}")
        print("=" * 100)
    elif level == 2:
        print("\n" + "-" * 100)
        print(f"  {title}")
        print("-" * 100)
    else:
        print(f"\n{'  ' * (level - 1)}► {title}")


def test_component_1_query_analyzer():
    """TEST 1: Query Analyzer - Extract intent, language, metadata"""
    print_section_header("TEST 1: QUERY ANALYZER", level=1)

    for idx, test_case in enumerate(TEST_QUESTIONS, 1):
        print_section_header(f"Question {idx}/{len(TEST_QUESTIONS)}", level=2)
        print(f"  Source: {test_case['source']}")
        print(f"  Language: {test_case['language']}")
        print(f"  Question: {test_case['question']}")

        try:
            analysis = query_analyzer.analyze(test_case["question"])
            print("\n  ANALYSIS RESULTS:")
            print(f"    • Intent: {analysis.intent}")
            print(f"    • Language: {analysis.language}")
            print(f"    • Doc Type: {analysis.doc_type or 'N/A'}")
            print(f"    • Section: {analysis.section_title or 'N/A'}")
            print(f"    • Chunk Type: {analysis.chunk_type or 'N/A'}")
            print(f"    • Is Procedural: {analysis.is_procedural}")
            print(f"    • Needs Warnings: {analysis.needs_warnings}")
            print(f"    • Confidence: {analysis.confidence}")
            print(f"    • Filters: {analysis.filters}")
            print("  ✓ SUCCESS")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")


def test_component_2_embedder():
    """TEST 2: Embedder - Text preparation strategy"""
    print_section_header("TEST 2: EMBEDDER - TEXT PREPARATION", level=1)

    test_chunks = [
        {
            "name": "Document Chunk (content)",
            "chunk": {
                "chunk_type": "content",
                "doc_title": "Health Insurance Policy",
                "section_title": "Coverage Details",
                "content": "The policy covers hospitalization up to 30 days with a maximum limit of $50,000 per year.",
            },
        },
        {
            "name": "FAQ Chunk",
            "chunk": {
                "chunk_type": "faq",
                "question": "What is the grace period?",
                "answer": "30 days from the due date for premium payments.",
            },
        },
        {
            "name": "Resolved Ticket Chunk",
            "chunk": {
                "chunk_type": "resolved_ticket",
                "question": "How to reset password?",
                "answer": "Request OTP via email and use it to set new password.",
            },
        },
    ]

    for test in test_chunks:
        print_section_header(test["name"], level=2)
        print(f"  Chunk Type: {test['chunk']['chunk_type']}")
        print(f"  Input Fields: {list(test['chunk'].keys())}")

        try:
            prepared_text = embedding_generator.prepare_text_for_embedding(
                test["chunk"]
            )
            print("\n  PREPARED TEXT FOR EMBEDDING:")
            print(f"    {prepared_text}")
            print("  ✓ SUCCESS")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")


def test_component_3_retriever():
    """TEST 3: LangChain Retriever - Hybrid search (BM25 + Dense + RRF)"""
    print_section_header("TEST 3: LANGCHAIN HYBRID RETRIEVER", level=1)

    if not langchain_retriever.is_ready():
        print("  ⚠ Retriever not initialized. Skipping test.")
        return

    # Test with 2 questions (1 English, 1 Arabic)
    test_queries = [TEST_QUESTIONS[0], TEST_QUESTIONS[1]]

    for idx, test_case in enumerate(test_queries, 1):
        print_section_header(f"Query {idx}: {test_case['question'][:60]}...", level=2)
        print(f"  Language: {test_case['language']}")

        try:
            chunks = langchain_retriever.search(
                query=test_case["question"], filters=None, top_k=5
            )

            print("\n  RETRIEVAL RESULTS:")
            print(f"    • Total chunks retrieved: {len(chunks)}")

            if chunks:
                print("\n  TOP 3 CHUNKS:")
                for i, chunk in enumerate(chunks[:3], 1):
                    content = chunk["payload"].get("content", "")[:100]
                    chunk_type = chunk["payload"].get("chunk_type", "unknown")
                    doc_id = chunk["payload"].get("doc_id", "unknown")
                    print(
                        f"\n    [{i}] Chunk ID: {chunk['payload'].get('chunk_id', 'N/A')}"
                    )
                    print(f"        Type: {chunk_type} | Doc: {doc_id}")
                    print(f"        Content: {content}...")

            print("  ✓ SUCCESS")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()


def test_component_4_reranker():
    """TEST 4: Jina AI Reranker - Score and rerank chunks"""
    print_section_header("TEST 4: JINA AI RERANKER", level=1)

    if not jina_reranker.is_available():
        print("  ⚠ Jina reranker not available. Skipping test.")
        return

    # Get chunks from retriever first
    test_case = TEST_QUESTIONS[0]  # Use first question
    print_section_header(f"Query: {test_case['question']}", level=2)

    try:
        # Retrieve chunks
        chunks = langchain_retriever.search(
            query=test_case["question"], filters=None, top_k=10
        )

        print(f"  Retrieved {len(chunks)} chunks before reranking")

        # Rerank
        reranked_chunks = jina_reranker.rerank(
            query=test_case["question"], chunks=chunks, top_n=5
        )

        print("\n  RERANKING RESULTS:")
        print(f"    • Reranked chunks: {len(reranked_chunks)}")

        if reranked_chunks:
            print("\n  TOP 3 RERANKED CHUNKS (with scores):")
            for i, chunk in enumerate(reranked_chunks[:3], 1):
                content = chunk["payload"].get("content", "")[:80]
                score = chunk.get("rerank_score", 0.0)
                print(f"\n    [{i}] Score: {score:.4f}")
                print(f"        Content: {content}...")

        print("  ✓ SUCCESS")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()


def test_component_5_context_builder():
    """TEST 5: Context Builder - Format context and extract citations/warnings/steps"""
    print_section_header("TEST 5: CONTEXT BUILDER", level=1)

    # Get chunks for a procedural query
    test_case = TEST_QUESTIONS[2]  # "How do I file an appeal?"
    print_section_header(f"Query: {test_case['question']}", level=2)

    try:
        # Get reranked chunks
        chunks = langchain_retriever.search(test_case["question"], top_k=10)
        reranked = jina_reranker.rerank(test_case["question"], chunks, top_n=5)

        # Build context
        context_data = context_builder.build_context_with_metadata(reranked)

        print("\n  CONTEXT BUILDING RESULTS:")
        print(f"    • Number of chunks: {context_data['num_chunks']}")
        print(f"    • Chunk types: {context_data['chunk_types']}")
        print(f"    • Citations extracted: {len(context_data['citations'])}")
        print(f"    • Warnings extracted: {len(context_data['warnings'])}")
        print(f"    • Steps extracted: {len(context_data['steps'])}")

        if context_data["citations"]:
            print("\n  CITATIONS:")
            for cite in context_data["citations"][:3]:
                print(f"    • {cite}")

        if context_data["warnings"]:
            print("\n  WARNINGS:")
            for warn in context_data["warnings"]:
                print(f"    • {warn}")

        if context_data["steps"]:
            print("\n  STEPS:")
            for step in context_data["steps"][:3]:
                print(f"    • {step}")

        print("\n  FORMATTED CONTEXT (first 300 chars):")
        print(f"    {context_data['formatted_context'][:300]}...")

        print("  ✓ SUCCESS")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()


async def test_component_6_end_to_end():
    """TEST 6: Full RAG Pipeline - Complete flow with all components"""
    print_section_header("TEST 6: END-TO-END RAG PIPELINE", level=1)

    for idx, test_case in enumerate(TEST_QUESTIONS, 1):
        print_section_header(f"Question {idx}/{len(TEST_QUESTIONS)}", level=2)
        print(f"  Source: {test_case['source']}")
        print(f"  Language: {test_case['language']}")
        print(f"  Question: {test_case['question']}")
        print(f"  Expected: {test_case['expected_answer']}")

        try:
            result = await new_rag_pipeline.process_query(
                query=test_case["question"], language=test_case["language"]
            )

            print("\n  PIPELINE RESULTS:")
            print(f"    • Answer length: {len(result['answer'])} chars")
            print(f"    • Citations: {len(result['citations'])}")
            print(f"    • Steps: {len(result['steps'])}")
            print(f"    • Warnings: {len(result['warnings'])}")
            print(f"    • Confidence: {result['confidence']:.2f}")
            print(f"    • Latency: {result['latency_ms']}ms")
            print(f"    • Token cost: {result['token_cost_estimate']}")

            print("\n  GENERATED ANSWER:")
            print(f"    {result['answer'][:400]}...")

            if result["citations"]:
                print("\n  CITATIONS:")
                for cite in result["citations"][:3]:
                    print(f"    • Doc: {cite['doc_id']}, Section: {cite['section']}")

            if result["steps"]:
                print("\n  STEPS EXTRACTED:")
                for i, step in enumerate(result["steps"][:3], 1):
                    print(f"    {i}. {step}")

            if result["warnings"]:
                print("\n  WARNINGS:")
                for warn in result["warnings"]:
                    print(f"    ⚠ {warn}")

            print("  ✓ SUCCESS")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()


async def main():
    """Run all component tests sequentially"""
    print("\n" + "=" * 100)
    print("  RAG PIPELINE COMPONENT TESTING")
    print("  Testing each component with fixed questions from FAQ and Resolved Tickets")
    print("=" * 100)

    # Initialize
    print("\n[INITIALIZATION]")
    print("  Loading chunks and initializing components...")

    try:
        chunks = vector_store.load_chunks_from_directory()
        print(f"  ✓ Loaded {len(chunks)} chunks")

        if not langchain_retriever.is_ready():
            langchain_retriever.initialize_bm25(chunks)
            langchain_retriever.initialize_vector_store()
            print("  ✓ LangChain retriever initialized")
        else:
            print("  ✓ LangChain retriever already ready")

    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        return

    # Run tests
    print("\n[RUNNING TESTS]\n")

    try:
        test_component_1_query_analyzer()
    except Exception as e:
        print(f"\nTest 1 crashed: {e}")

    try:
        test_component_2_embedder()
    except Exception as e:
        print(f"\nTest 2 crashed: {e}")

    try:
        test_component_3_retriever()
    except Exception as e:
        print(f"\nTest 3 crashed: {e}")

    try:
        test_component_4_reranker()
    except Exception as e:
        print(f"\nTest 4 crashed: {e}")

    try:
        test_component_5_context_builder()
    except Exception as e:
        print(f"\nTest 5 crashed: {e}")

    try:
        await test_component_6_end_to_end()
    except Exception as e:
        print(f"\nTest 6 crashed: {e}")

    # Summary
    print("\n" + "=" * 100)
    print("  TESTING COMPLETE")
    print("  Review outputs above to assess component performance and accuracy")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
