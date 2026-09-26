from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="allow")
    case_id: str
    user_id: str = "eval-user"
    session_id: str | None = None
    source_paths: list[str] = Field(default_factory=list)
    selected_source_ids: list[str] = Field(default_factory=list)
    query: str = ""
    request_mode: str = "auto"
    top_k: int | None = None
    transformation_config: dict[str, Any] = Field(default_factory=dict)
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    reference_text: str | None = None
    expected_formats: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class EvalDataset(BaseModel):
    name: str = "evaluation"
    version: str = "1.0"
    cases: list[EvalCase] = Field(default_factory=list)


class EvalCaseResult(BaseModel):
    case_id: str
    status: str
    passed: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    dataset_name: str
    dataset_version: str
    cases: list[EvalCaseResult]
    aggregate: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
