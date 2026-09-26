from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from generation.crr import TransformationConfig
from retrieval.config import RetrievalConfig


class TaskSpec(BaseModel):
    """Generic service-level request contract without changing current intent routing.

    `request_mode` deliberately remains auto/qa/transform for this prototype. The
    object standardizes clients now while allowing a richer operation registry later.
    """

    model_config = ConfigDict(extra="allow")

    user_id: str
    session_id: str | None = None
    source_paths: list[str] = Field(default_factory=list)
    selected_source_ids: list[str] = Field(default_factory=list)
    query: str = ""
    request_mode: str = "auto"
    session_title: str | None = None
    retrieval: RetrievalConfig | None = None
    transformation: TransformationConfig = Field(default_factory=TransformationConfig)
