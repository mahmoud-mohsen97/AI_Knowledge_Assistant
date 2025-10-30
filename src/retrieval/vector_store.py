#!/usr/bin/env python3
"""
Qdrant vector store integration and indexing
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)
from src.config.settings import config
from src.retrieval.embedder import embedding_generator
from src.utils.logger import logger
from src.utils.helpers import extract_chunk_content


class VectorStore:
    """Qdrant vector store wrapper"""

    def __init__(self):
        """Initialize Qdrant client and collection"""
        self.client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            api_key=config.QDRANT_API_KEY,
        )
        self.collection_name = config.QDRANT_COLLECTION_NAME
        self.embedding_dim = config.EMBEDDING_DIMENSION
        logger.info(
            f"Initialized VectorStore connecting to {config.QDRANT_HOST}:{config.QDRANT_PORT}"
        )

    def collection_exists(self) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections().collections
            return any(col.name == self.collection_name for col in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def create_collection(self):
        """Create Qdrant collection with proper configuration"""
        try:
            if self.collection_exists():
                logger.info(f"Collection '{self.collection_name}' already exists")
                return

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, distance=Distance.COSINE
                ),
            )
            logger.info(f"Created collection '{self.collection_name}'")

            # Create payload indexes for efficient filtering
            self._create_payload_indexes()
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def _create_payload_indexes(self):
        """Create indexes on frequently filtered fields"""
        fields_to_index = [
            "doc_id",
            "doc_type",
            "chunk_type",
            "language",
            "region",
            "section_title",
        ]

        for field in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
                logger.info(f"Created payload index for field: {field}")
            except Exception as e:
                logger.warning(f"Could not create index for {field}: {e}")

    def get_collection_count(self) -> int:
        """Get number of points in collection"""
        try:
            if not self.collection_exists():
                return 0

            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0

    def is_empty(self) -> bool:
        """Check if collection is empty"""
        return self.get_collection_count() == 0

    def load_chunks_from_directory(
        self, chunks_dir: Path = None
    ) -> List[Dict[str, Any]]:
        """
        Load all chunks from directory

        Args:
            chunks_dir: Directory containing chunk JSON files

        Returns:
            List of chunk dictionaries
        """
        chunks_dir = chunks_dir or config.PROCESSED_CHUNKS_DIR
        chunks = []

        if not chunks_dir.exists():
            logger.error(f"Chunks directory not found: {chunks_dir}")
            return chunks

        json_files = list(chunks_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} chunk files")

        for json_file in json_files:
            # Skip all_chunks.json to avoid loading duplicates
            # (it contains all chunks that are also in individual files)
            if json_file.name == "all_chunks.json":
                logger.info(
                    f"Skipping {json_file.name} (combined file, using individual chunks instead)"
                )
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Handle both individual chunk files and any other list format
                    if isinstance(data, list):
                        # This is a list of chunks (shouldn't happen with individual files, but handle it)
                        logger.info(f"{json_file.name} contains {len(data)} chunks")
                        chunks.extend(data)
                    elif isinstance(data, dict):
                        # This is a single chunk file (expected format)
                        chunks.append(data)
                    else:
                        logger.warning(
                            f"Unexpected data type in {json_file.name}: {type(data)}"
                        )
            except Exception as e:
                logger.warning(f"Error loading {json_file.name}: {e}")

        logger.info(f"Loaded {len(chunks)} chunks total")
        return chunks

    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Index chunks into Qdrant

        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for processing
        """
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info(f"Starting indexing of {len(chunks)} chunks")

        # Extract content for embedding
        texts = []
        for chunk in chunks:
            content = extract_chunk_content(chunk)
            texts.append(content)

        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        embeddings = embedding_generator.generate_embeddings_batch(
            texts, batch_size=batch_size
        )

        # Create points for Qdrant
        logger.info("Creating Qdrant points...")
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Prepare payload with all metadata
            payload = {
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "doc_id": chunk.get("doc_id", ""),
                "doc_title": chunk.get("doc_title", ""),
                "doc_summary": chunk.get("doc_summary", ""),
                "doc_type": chunk.get("doc_type", ""),
                "chunk_type": chunk.get("chunk_type", "text"),
                "content": extract_chunk_content(chunk),
                "page_num": chunk.get("page_num"),
                "section_title": chunk.get("section_title", ""),
                "language": chunk.get("language", "en"),
                "region": chunk.get("region", ""),
                "version": chunk.get("version", ""),
                "source": chunk.get("source", ""),
                "image_path": chunk.get("image_path", ""),
            }

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            point = PointStruct(id=idx, vector=embedding, payload=payload)
            points.append(point)

        # Upload points in batches
        logger.info("Uploading points to Qdrant...")
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            try:
                self.client.upsert(collection_name=self.collection_name, points=batch)
                logger.info(
                    f"Uploaded batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}"
                )
            except Exception as e:
                logger.error(f"Error uploading batch: {e}")
                raise

        logger.info(f"Successfully indexed {len(chunks)} chunks")

    def search(
        self,
        query_vector: List[float],
        limit: int = 20,
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding vector
            limit: Number of results to return
            filters: Qdrant filters

        Returns:
            List of search results with payload and score
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filters,
            )

            return [
                {"id": result.id, "score": result.score, "payload": result.payload}
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def build_filter(self, filter_dict: Dict[str, Any]) -> Optional[Filter]:
        """
        Build Qdrant filter from dictionary

        Args:
            filter_dict: Dictionary with filter conditions

        Returns:
            Qdrant Filter object
        """
        if not filter_dict:
            return None

        conditions = []

        for key, value in filter_dict.items():
            if value is None:
                continue

            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=conditions) if conditions else None

    def initialize_and_index_if_empty(self):
        """
        Initialize collection and index chunks if empty
        This is called on startup
        """
        try:
            # Create collection if it doesn't exist
            if not self.collection_exists():
                logger.info("Collection doesn't exist, creating...")
                self.create_collection()

            # Check if empty and index
            if self.is_empty():
                logger.info("Collection is empty, loading and indexing chunks...")
                chunks = self.load_chunks_from_directory()
                if chunks:
                    self.index_chunks(chunks)
                else:
                    logger.warning("No chunks found to index")
            else:
                count = self.get_collection_count()
                logger.info(f"Collection already has {count} points")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise


# Singleton instance
vector_store = VectorStore()
