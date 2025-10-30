#!/usr/bin/env python3
"""
LangChain-based hybrid retrieval system combining BM25 and dense vector search
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config.settings import config
from src.retrieval.embedder import embedding_generator
from src.utils.logger import logger


class LangChainRetriever:
    """Hybrid retriever using LangChain's BM25 and Qdrant integration"""

    def __init__(self):
        """Initialize LangChain-based retriever"""
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL, openai_api_key=config.OPENAI_API_KEY
        )

        self.qdrant_client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            api_key=config.QDRANT_API_KEY,
        )

        self.collection_name = config.QDRANT_COLLECTION_NAME
        self.bm25_retriever = None
        self.vector_store = None
        self.ensemble_retriever = None

        logger.info("Initialized LangChainRetriever")

    def initialize_vector_store(self):
        """Initialize Qdrant vector store"""
        try:
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            logger.info("Qdrant vector store initialized")
        except Exception as e:
            logger.error(f"Error initializing Qdrant vector store: {e}")
            raise

    def initialize_bm25(self, chunks: List[Dict[str, Any]]):
        """
        Initialize BM25 retriever with chunks

        Args:
            chunks: List of chunk dictionaries
        """
        logger.info(f"Initializing BM25 retriever with {len(chunks)} chunks")

        try:
            # Convert chunks to LangChain Documents
            documents = []
            for chunk in chunks:
                # Prepare text using the same strategy as embedder
                text = embedding_generator.prepare_text_for_embedding(chunk)

                # Create Document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "chunk_id": chunk.get("chunk_id", ""),
                        "doc_id": chunk.get("doc_id", ""),
                        "doc_type": chunk.get("doc_type", ""),
                        "section_title": chunk.get("section_title", ""),
                        "chunk_type": chunk.get("chunk_type", "content"),
                        "language": chunk.get("language", "en"),
                        "region": chunk.get("region", ""),
                        "page_num": chunk.get("page_num"),
                        # Store original chunk for later use
                        "_original_chunk": chunk,
                    },
                )
                documents.append(doc)

            # Create BM25 retriever
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = config.RETRIEVAL_TOP_K

            logger.info("BM25 retriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BM25 retriever: {e}")
            raise

    def _reciprocal_rank_fusion(
        self, bm25_docs: List[Document], dense_docs: List[Document], k: int = 60
    ) -> List[Document]:
        """
        Combine BM25 and dense results using Reciprocal Rank Fusion

        Args:
            bm25_docs: Documents from BM25 retrieval
            dense_docs: Documents from dense retrieval
            k: RRF constant (default: 60)

        Returns:
            Fused and ranked documents
        """
        rrf_scores = defaultdict(float)
        doc_map = {}

        # Process BM25 results
        for rank, doc in enumerate(bm25_docs, start=1):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            rrf_scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

        # Process dense results
        for rank, doc in enumerate(dense_docs, start=1):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            rrf_scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

        # Sort by RRF score
        sorted_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return documents in RRF order
        return [doc_map[doc_id] for doc_id, _ in sorted_doc_ids if doc_id in doc_map]

    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """
        Build Qdrant filter from metadata filters

        Args:
            filters: Dictionary of filter conditions

        Returns:
            Qdrant Filter object
        """
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            if value is not None and value != "":
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        if not conditions:
            return None

        return Filter(must=conditions)

    def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and dense retrieval

        Args:
            query: Search query
            filters: Metadata filters
            top_k: Number of results to return

        Returns:
            List of retrieved chunks
        """
        top_k = top_k or config.RETRIEVAL_TOP_K

        logger.info(f"Performing hybrid search for query: '{query[:100]}...'")

        try:
            if not self.vector_store:
                self.initialize_vector_store()

            # 1. Dense retrieval (vector search)
            search_kwargs = {"k": top_k}
            if filters:
                qdrant_filter = self._build_qdrant_filter(filters)
                if qdrant_filter:
                    search_kwargs["filter"] = qdrant_filter

            dense_retriever = self.vector_store.as_retriever(
                search_kwargs=search_kwargs
            )
            dense_docs = dense_retriever.invoke(
                query
            )  # Changed from get_relevant_documents
            logger.debug(f"Dense retrieval: {len(dense_docs)} documents")

            # 2. BM25 retrieval (sparse)
            bm25_docs = []
            if self.bm25_retriever:
                self.bm25_retriever.k = top_k
                bm25_docs = self.bm25_retriever.invoke(
                    query
                )  # Changed from get_relevant_documents
                logger.debug(f"BM25 retrieval: {len(bm25_docs)} documents")

            # 3. Reciprocal Rank Fusion
            if bm25_docs:
                fused_docs = self._reciprocal_rank_fusion(bm25_docs, dense_docs)
                logger.info(f"RRF fusion: {len(fused_docs)} unique documents")
            else:
                fused_docs = dense_docs
                logger.info("Using only dense results (BM25 not available)")

            # 4. Convert to chunk format
            chunks = []
            for doc in fused_docs[:top_k]:
                original_chunk = doc.metadata.get("_original_chunk")
                if original_chunk:
                    chunk = {"payload": original_chunk, "score": 1.0}
                else:
                    chunk = {
                        "payload": {
                            "chunk_id": doc.metadata.get("chunk_id", ""),
                            "doc_id": doc.metadata.get("doc_id", ""),
                            "doc_type": doc.metadata.get("doc_type", ""),
                            "section_title": doc.metadata.get("section_title", ""),
                            "chunk_type": doc.metadata.get("chunk_type", "content"),
                            "content": doc.page_content,
                            "language": doc.metadata.get("language", "en"),
                            "region": doc.metadata.get("region", ""),
                            "page_num": doc.metadata.get("page_num"),
                        },
                        "score": 1.0,
                    }
                chunks.append(chunk)

            logger.info(f"Hybrid search returned {len(chunks)} results")
            return chunks

        except Exception as e:
            logger.error(f"Error during hybrid search: {e}", exc_info=True)
            return []

    def is_ready(self) -> bool:
        """Check if retriever is ready"""
        return self.bm25_retriever is not None or self.vector_store is not None


# Singleton instance
langchain_retriever = LangChainRetriever()
