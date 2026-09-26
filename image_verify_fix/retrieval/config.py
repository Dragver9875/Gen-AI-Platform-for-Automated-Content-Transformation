from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    strategy: Literal["hybrid", "dense", "lexical"] = "hybrid"
    bm25_k: int = Field(default=15, ge=1, le=500)
    vector_k: int = Field(default=15, ge=1, le=500)
    final_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    use_reranker: bool = True
