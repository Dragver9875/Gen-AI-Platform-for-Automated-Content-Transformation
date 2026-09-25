from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.nodes import Phase3Nodes
from agents.state import AgentState
from app.phase12 import Phase12Pipeline


def _route_after_initialize(state: AgentState) -> str:
    if state.get("status") == "error":
        return "error"
    return "ingest" if state.get("pending_source_paths") else "classify"


def _route_intent(state: AgentState) -> str:
    intent = state.get("detected_intent")
    if intent == "qa":
        return "qa"
    if intent == "transform":
        return "transform"
    if intent == "index_only":
        return "index_only"
    return "error"


def build_phase3_graph(
    pipeline: Phase12Pipeline,
    *,
    default_top_k: int = 5,
    context_max_chars: int = 60000,
    checkpointer: Any | None = None,
):
    """Build and compile the Phase 3 LangGraph.

    If no checkpointer is supplied an InMemorySaver is used so thread/session
    continuation works in development and tests. Production deployments should
    inject a durable checkpointer (e.g. Postgres) or rely on the LangGraph
    deployment platform's persistence layer.
    """
    nodes = Phase3Nodes(
        pipeline,
        default_top_k=default_top_k,
        context_max_chars=context_max_chars,
    )

    builder = StateGraph(AgentState)
    builder.add_node("initialize", nodes.initialize)
    builder.add_node("ingest_sources", nodes.ingest_sources)
    builder.add_node("classify_intent", nodes.classify_intent)
    builder.add_node("retrieve_qa", nodes.retrieve_qa)
    builder.add_node("retrieve_transform", nodes.retrieve_transform)
    builder.add_node("build_context", nodes.build_context)
    builder.add_node("finalize_index_only", nodes.finalize_index_only)
    builder.add_node("finalize_error", nodes.finalize_error)

    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        _route_after_initialize,
        {
            "ingest": "ingest_sources",
            "classify": "classify_intent",
            "error": "finalize_error",
        },
    )
    builder.add_edge("ingest_sources", "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "qa": "retrieve_qa",
            "transform": "retrieve_transform",
            "index_only": "finalize_index_only",
            "error": "finalize_error",
        },
    )
    builder.add_edge("retrieve_qa", "build_context")
    builder.add_edge("retrieve_transform", "build_context")
    builder.add_edge("build_context", END)
    builder.add_edge("finalize_index_only", END)
    builder.add_edge("finalize_error", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver(), name="phase3-orchestrator")


class Phase3Orchestrator:
    """Thin invocation façade around the compiled LangGraph."""

    def __init__(self, graph, *, default_top_k: int = 5):
        self.graph = graph
        self.default_top_k = default_top_k

    def invoke(
        self,
        *,
        session_id: str,
        query: str | None = None,
        source_paths: list[str] | None = None,
        request_mode: str = "auto",
        selected_source_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> AgentState:
        # Explicitly clear per-invocation fields so checkpointed values from the
        # previous turn do not accidentally retrigger ingestion or an old query.
        payload: AgentState = {
            "session_id": session_id,
            "query": query or "",
            "pending_source_paths": list(source_paths or []),
            "request_mode": request_mode,  # type: ignore[typeddict-item]
            "selected_source_ids": list(selected_source_ids or []),
            "top_k": int(top_k or self.default_top_k),
            "retrieved_documents": [],
            "context_groups": [],
            "prepared_context": "",
            "warnings": [],
            "errors": [],
        }
        config = {"configurable": {"thread_id": session_id}}
        return self.graph.invoke(payload, config=config)

    def get_state(self, session_id: str):
        return self.graph.get_state({"configurable": {"thread_id": session_id}})
