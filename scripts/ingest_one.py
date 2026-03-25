#!/usr/bin/env python3
"""
Ingest a single ADP trajectory into Qdrant.

Usage:
    python scripts/ingest_one.py data/adp/d5f22c3e6cc74fc08a12420c97b405f8.json

Prerequisites:
    1. Qdrant running: docker run -p 6333:6333 qdrant/qdrant
    2. Dependencies:   pip install sentence-transformers qdrant-client pydantic-settings
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recall.server.ingest import main

if __name__ == "__main__":
    main()
