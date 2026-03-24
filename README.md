# Recall

A vector-based trajectory storage and retrieval system for AI agent execution traces. Recall indexes agent sessions in a semantic vector database (Qdrant), enabling similarity search over past executions to surface relevant context for future runs.

## Overview

Recall converts agent interaction sequences into a standardized **ADP (Agent Data Protocol)** format, embeds them using sentence transformers, and stores them in Qdrant. You can then query by natural language to find trajectories from similar past tasks.

**Current support:** OpenHands V1 session recordings.

## Architecture

```
OpenHands sessions
      │
      ▼
  to_adp.py          Convert raw events → ADP JSON
      │
      ▼
  ingest.py          Embed + upsert into Qdrant
      │
      ▼
  qdrant_store.py    Vector storage (cosine similarity)
      │
      ▼
  retrieve.py        Semantic search (WIP)
```

## Project Structure

```
recall/
├── recall/
│   ├── config/settings.py          # Pydantic-based config (env vars)
│   ├── embeddings/embedder.py      # Sentence-Transformers wrapper
│   ├── storage/qdrant_store.py     # Qdrant client (upsert, search)
│   ├── server/
│   │   ├── ingest.py               # Ingestion pipeline
│   │   ├── retrieve.py             # Retrieval (WIP)
│   │   ├── inject.py               # Context injection (WIP)
│   │   └── rerank.py               # Reranking (WIP)
│   └── wrappers/openhands/
│       └── to_adp.py               # OpenHands V1 → ADP converter
├── scripts/
│   ├── ingest_one.py               # Ingest a single ADP file
│   └── create_collections.py       # Qdrant collection setup (WIP)
└── data/
    ├── adp/                        # Converted ADP trajectory JSONs
    └── raw/                        # Raw source data
```

## Requirements

- Python 3.10+
- [Qdrant](https://qdrant.tech/) vector database
- `sentence-transformers`
- `qdrant-client`
- `pydantic-settings`

## Setup

**1. Start Qdrant:**

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**2. Install dependencies:**

```bash
pip install sentence-transformers qdrant-client pydantic-settings
```

**3. Configure environment** (`.env` or shell env):

```env
RECALL_QDRANT_URL=http://localhost:6333
RECALL_QDRANT_COLLECTION=trajectories
RECALL_EMBEDDING_MODEL=all-MiniLM-L6-v2
RECALL_ADP_DIR=data/adp
RECALL_RAW_DIR=data/raw
RECALL_OPENHANDS_CONVERSATIONS_DIR=~/.openhands/conversations
```

## Usage

**Convert an OpenHands session to ADP format:**

```bash
python -m recall.wrappers.openhands.to_adp \
  ~/.openhands/conversations/<session_id>/events \
  --output data/adp/<session_id>.json
```

**Ingest a single ADP trajectory into Qdrant:**

```bash
python scripts/ingest_one.py data/adp/<session_id>.json
```

**Search for similar trajectories (programmatic):**

```python
from recall.embeddings.embedder import Embedder
from recall.storage.qdrant_store import QdrantStore

embedder = Embedder()
store = QdrantStore()

query_vec = embedder.embed("fix authentication bug in login flow")
results = store.search(query_vec, limit=5)

for r in results:
    print(r["payload"]["task_goal"], r["score"])
```

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `RECALL_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `RECALL_QDRANT_COLLECTION` | `trajectories` | Qdrant collection name |
| `RECALL_QDRANT_API_KEY` | `None` | API key (for Qdrant Cloud) |
| `RECALL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `RECALL_EMBEDDING_DIM` | `384` | Vector dimension |
| `RECALL_DATA_DIR` | `data` | Base data directory |
| `RECALL_ADP_DIR` | `data/adp` | ADP trajectory storage |
| `RECALL_RAW_DIR` | `data/raw` | Raw input data |
| `RECALL_OPENHANDS_CONVERSATIONS_DIR` | `~/.openhands/conversations` | OpenHands session directory |

## Status

| Component | Status |
|---|---|
| OpenHands → ADP conversion | Done |
| Embedding + Qdrant ingestion | Done |
| Vector search | WIP |
| REST API | WIP |
| Context injection | WIP |
| Reranking | WIP |
| Evaluation | WIP |
