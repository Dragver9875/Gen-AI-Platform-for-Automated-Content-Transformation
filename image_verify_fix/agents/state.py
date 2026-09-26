from __future__ import annotations

from typing import Any, Literal, TypedDict


RequestMode = Literal["auto", "qa", "transform"]
DetectedIntent = Literal["index_only", "qa", "transform", "error"]


class AgentState(TypedDict, total=False):
    """Serializable state shared by the Phase 3 LangGraph.

    user_id + session_id are the multi-tenant isolation boundary. Provider/model
    objects never enter this state. Only JSON-compatible values are checkpointed.
    """

    # User/session/request identity
    user_id: str
    session_id: str
    query: str
    request_mode: RequestMode
    top_k: int

    pending_source_paths: list[str]
    active_source_ids: list[str]
    selected_source_ids: list[str]
    ingested_sources: list[dict[str, Any]]

    detected_intent: DetectedIntent
    intent_confidence: float
    intent_reason: str
    retrieval_mode: str
    retrieval_config: dict[str, Any]

    retrieved_documents: list[dict[str, Any]]
    context_groups: list[dict[str, Any]]
    prepared_context: str

    # Phase 4 generation
    transformation_config: dict[str, Any]
    canonical_response: dict[str, Any]
    section_digests: list[dict[str, Any]]
    generation_metadata: dict[str, Any]

    # Phase 5 factuality verification / repair
    verification_report: dict[str, Any]
    verification_metadata: dict[str, Any]
    verification_requires_repair: bool
    repair_attempts: int
    max_repair_attempts: int

    # Phase 6 artifact generation
    artifacts: list[dict[str, Any]]
    artifact_metadata: dict[str, Any]

    status: str
    warnings: list[str]
    errors: list[str]
