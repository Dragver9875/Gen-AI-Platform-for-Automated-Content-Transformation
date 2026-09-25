from __future__ import annotations

from typing import Any, Literal, TypedDict


RequestMode = Literal["auto", "qa", "transform"]
DetectedIntent = Literal["index_only", "qa", "transform", "error"]


class AgentState(TypedDict, total=False):
    """Serializable state shared by the Phase 3 LangGraph.

    Model/provider objects never enter this state. Only JSON-compatible values are
    stored so the same graph can later use a persistent production checkpointer.
    """

    # Session/request identity
    session_id: str
    query: str
    request_mode: RequestMode
    top_k: int

    # New files supplied on this invocation. The ingestion node clears this list
    # after processing so resumed threads never re-index stale paths.
    pending_source_paths: list[str]

    # Sources indexed during the lifetime of the LangGraph thread.
    active_source_ids: list[str]
    selected_source_ids: list[str]
    ingested_sources: list[dict[str, Any]]

    # Routing
    detected_intent: DetectedIntent
    intent_confidence: float
    intent_reason: str
    retrieval_mode: str

    # Retrieval/context handoff to Phase 4
    retrieved_documents: list[dict[str, Any]]
    context_groups: list[dict[str, Any]]
    prepared_context: str

    # Operational state
    status: str
    warnings: list[str]
    errors: list[str]
