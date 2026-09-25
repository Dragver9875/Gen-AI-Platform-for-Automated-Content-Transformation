"""Backward-compatible database helper.

Phase 1 is cloud-only: no local Chroma persistence fallback is used.
"""
from __future__ import annotations

import os


def create_chroma_client(*_args, **_kwargs):
    missing = [name for name in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE") if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Chroma Cloud configuration missing: {', '.join(missing)}")
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required for Chroma Cloud. Install requirements.txt") from exc
    return chromadb.CloudClient()
