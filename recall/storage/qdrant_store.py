"""
Qdrant storage — handles collection management and point operations.

This is the only module that talks to Qdrant directly.
Everything else goes through this interface.

Usage:
    from recall.storage.qdrant_store import QdrantStore

    store = QdrantStore()
    store.ensure_collection()
    store.upsert_trajectory(trajectory_id, vector, payload)
    results = store.search(query_vector, limit=5)
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger("recall.qdrant")


class QdrantStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "trajectories",
        embedding_dim: int = 384,
        api_key: str | None = None,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            logger.info("Collection '%s' already exists", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection '%s' (dim=%d, cosine)",
            self.collection_name,
            self.embedding_dim,
        )

    def upsert_trajectory(
        self,
        trajectory_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """
        Upsert a single trajectory into Qdrant.

        Parameters
        ----------
        trajectory_id
            The ADP trajectory ID (session ID). Used as the Qdrant point ID
            via a deterministic hash, making upserts idempotent.
        vector
            The embedding vector for this trajectory.
        payload
            Metadata stored alongside the vector. Typically includes:
            - task_goal: str
            - tools_used: list[str]
            - error_count: int
            - total_steps: int
            - started_at: str
            - ended_at: str
            - working_dir: str
            - source_format: str
            - adp_content: list[dict]  (the full trajectory content)
        """
        # Qdrant accepts string IDs (UUID format) or integer IDs.
        # We use the trajectory_id directly as a string point ID.
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=trajectory_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        logger.info("Upserted trajectory %s", trajectory_id)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar trajectories.

        Returns a list of dicts with keys: id, score, payload
        """
        qdrant_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    def count(self) -> int:
        """Return the number of points in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def delete_trajectory(self, trajectory_id: str) -> None:
        """Delete a trajectory by ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[trajectory_id],
        )
        logger.info("Deleted trajectory %s", trajectory_id)
