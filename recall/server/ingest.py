"""
Ingestion pipeline — reads ADP trajectory JSON, embeds it, stores in Qdrant.

This is the core "transmitter" that gets trajectories into the vector DB.

Usage:
    from recall.server.ingest import ingest_trajectory_file, ingest_directory

    # Single file
    ingest_trajectory_file("data/adp/d5f22c3e6cc74fc08a12420c97b405f8.json")

    # All files in a directory
    ingest_directory("data/adp/")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("recall.ingest")


def _build_embedding_text(trajectory: dict[str, Any]) -> str:
    """
    Build the text that gets embedded for semantic search.

    Strategy: combine the task goal with a summary of the actions taken.
    This gives the vector both "what was the task" and "how was it solved"
    semantics, so retrieval works whether you search by goal or by approach.
    """
    details = trajectory.get("details", {})
    parts = []

    # Task goal (most important signal)
    goal = details.get("task_goal", "")
    if goal:
        parts.append(f"Task: {goal}")

    # Tools and domain context
    tools = details.get("tools_available", [])
    if tools:
        parts.append(f"Tools: {', '.join(tools)}")

    working_dir = details.get("working_dir", "")
    if working_dir:
        parts.append(f"Project: {working_dir.split('/')[-1]}")

    # Action summary — first few actions to capture the approach
    content = trajectory.get("content", [])
    action_summaries = []
    for item in content:
        cls = item.get("class_", "")
        if cls == "code_action":
            cmd = item.get("content", "")[:100]
            action_summaries.append(f"bash: {cmd}")
        elif cls == "api_action":
            func = item.get("function", "")
            kwargs = item.get("kwargs", {})
            path = kwargs.get("path", "")
            if path:
                action_summaries.append(f"{func}({path.split('/')[-1]})")
            else:
                action_summaries.append(func)
        elif cls == "message_action":
            msg = item.get("content", "")[:80]
            action_summaries.append(f"reply: {msg}")

        if len(action_summaries) >= 8:  # cap to keep embedding text focused
            break

    if action_summaries:
        parts.append("Actions: " + " → ".join(action_summaries))

    # Error context
    error_count = details.get("error_count", 0)
    if error_count > 0:
        parts.append(f"Errors encountered: {error_count}")

    return "\n".join(parts)


def _build_payload(trajectory: dict[str, Any]) -> dict[str, Any]:
    """
    Build the Qdrant payload — metadata stored alongside the vector.

    We store both the searchable metadata (for filtering) and the
    full ADP content (for injection when retrieved).
    """
    details = trajectory.get("details", {})

    return {
        # Searchable / filterable fields
        "task_goal": details.get("task_goal", ""),
        "tools_available": details.get("tools_available", []),
        "working_dir": details.get("working_dir", ""),
        "error_count": details.get("error_count", 0),
        "total_events": details.get("total_events", 0),
        "started_at": details.get("started_at", ""),
        "ended_at": details.get("ended_at", ""),
        "source_format": details.get("source_format", ""),
        "action_counts": details.get("action_counts", {}),

        # The full ADP content — used for injection during retrieval
        "adp_content": trajectory.get("content", []),

        # Number of ADP items (useful for quick filtering)
        "adp_item_count": len(trajectory.get("content", [])),
    }


def ingest_trajectory_file(
    adp_json_path: str | Path,
    embedder=None,
    store=None,
) -> str:
    """
    Ingest a single ADP trajectory JSON file into Qdrant.

    Parameters
    ----------
    adp_json_path
        Path to the ADP trajectory JSON file (output of to_adp.py).
    embedder
        Optional Embedder instance. Created from settings if not provided.
    store
        Optional QdrantStore instance. Created from settings if not provided.

    Returns
    -------
    The trajectory ID that was ingested.
    """
    # Lazy imports to avoid loading models at module import time
    if embedder is None:
        from recall.embeddings.embedder import Embedder
        from recall.config.settings import get_settings
        settings = get_settings()
        embedder = Embedder(settings.embedding_model)

    if store is None:
        from recall.storage.qdrant_store import QdrantStore
        from recall.config.settings import get_settings
        settings = get_settings()
        store = QdrantStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            embedding_dim=settings.embedding_dim,
            api_key=settings.qdrant_api_key,
        )
        store.ensure_collection()

    # Read the trajectory
    adp_json_path = Path(adp_json_path)
    with open(adp_json_path) as f:
        trajectory = json.load(f)

    trajectory_id = trajectory["id"]

    # Build embedding text and embed it
    embedding_text = _build_embedding_text(trajectory)
    logger.debug("Embedding text for %s:\n%s", trajectory_id, embedding_text[:200])
    vector = embedder.embed(embedding_text)

    # Build payload and upsert
    payload = _build_payload(trajectory)
    store.upsert_trajectory(trajectory_id, vector, payload)

    logger.info(
        "Ingested %s: %d ADP items, vector dim=%d",
        trajectory_id,
        payload["adp_item_count"],
        len(vector),
    )
    return trajectory_id


def ingest_directory(
    adp_dir: str | Path,
    embedder=None,
    store=None,
) -> list[str]:
    """
    Ingest all ADP trajectory JSON files from a directory.

    Returns list of ingested trajectory IDs.
    """
    from recall.embeddings.embedder import Embedder
    from recall.storage.qdrant_store import QdrantStore
    from recall.config.settings import get_settings

    settings = get_settings()

    if embedder is None:
        embedder = Embedder(settings.embedding_model)
    if store is None:
        store = QdrantStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            embedding_dim=settings.embedding_dim,
            api_key=settings.qdrant_api_key,
        )
        store.ensure_collection()

    adp_dir = Path(adp_dir)
    json_files = sorted(adp_dir.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", adp_dir)
        return []

    ingested = []
    for jf in json_files:
        try:
            tid = ingest_trajectory_file(jf, embedder=embedder, store=store)
            ingested.append(tid)
        except Exception as e:
            logger.error("Failed to ingest %s: %s", jf.name, e)

    logger.info("Ingested %d / %d trajectories", len(ingested), len(json_files))
    return ingested


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingest ADP trajectories into Qdrant")
    parser.add_argument(
        "path",
        help="Path to a single ADP JSON file or a directory of JSON files",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--collection",
        default="trajectories",
        help="Qdrant collection name (default: trajectories)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    # Import here to avoid slow model loading if --help is called
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from recall.embeddings.embedder import Embedder
    from recall.storage.qdrant_store import QdrantStore

    embedder = Embedder(args.model)
    store = QdrantStore(
        url=args.qdrant_url,
        collection_name=args.collection,
        embedding_dim=embedder.dim,
    )
    store.ensure_collection()

    target = Path(args.path)

    if target.is_file():
        tid = ingest_trajectory_file(target, embedder=embedder, store=store)
        print(f"\nIngested: {tid}")
    elif target.is_dir():
        tids = ingest_directory(target, embedder=embedder, store=store)
        print(f"\nIngested {len(tids)} trajectories")
        for tid in tids:
            print(f"  {tid}")
    else:
        print(f"Error: {target} is not a file or directory")
        return

    print(f"\nCollection '{args.collection}' now has {store.count()} points")


if __name__ == "__main__":
    main()
