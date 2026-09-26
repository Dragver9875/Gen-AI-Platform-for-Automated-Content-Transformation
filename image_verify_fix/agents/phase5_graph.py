from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.graph import _route_after_initialize, _route_intent
from agents.nodes import Phase3Nodes
from agents.phase4_graph import _route_generation, Phase4Orchestrator
from agents.phase4_nodes import Phase4Nodes
from agents.phase5_nodes import Phase5Nodes
from agents.state import AgentState
from app.phase12 import Phase12Pipeline
from app.sessions import UserSessionManager, make_thread_id
from generation.crr import TransformationConfig
from generation.service import GenerationService
from verification.service import VerificationService


def _route_after_generation(state: AgentState) -> str:
    return "error" if state.get("status") == "error" else "verify"


def _route_verification(state: AgentState) -> str:
    if state.get("status") == "error":
        return "error"
    report = dict(state.get("verification_report") or {})
    if report.get("passed") is True:
        return "pass"
    attempts = int(state.get("repair_attempts") or 0)
    maximum = int(state.get("max_repair_attempts") or 0)
    if state.get("verification_requires_repair") and attempts < maximum:
        return "repair"
    return "issues"


def build_phase5_graph(
    pipeline: Phase12Pipeline,
    generator: GenerationService,
    verifier: VerificationService,
    *,
    default_top_k: int = 5,
    context_max_chars: int = 60000,
    checkpointer: Any | None = None,
):
    phase3 = Phase3Nodes(pipeline, default_top_k=default_top_k, context_max_chars=context_max_chars)
    phase4 = Phase4Nodes(generator)
    phase5 = Phase5Nodes(verifier)

    builder = StateGraph(AgentState)
    builder.add_node("initialize", phase3.initialize)
    builder.add_node("ingest_sources", phase3.ingest_sources)
    builder.add_node("classify_intent", phase3.classify_intent)
    builder.add_node("retrieve_qa", phase3.retrieve_qa)
    builder.add_node("retrieve_transform", phase3.retrieve_transform)
    builder.add_node("build_context", phase3.build_context)
    builder.add_node("generate_qa", phase4.generate_qa)
    builder.add_node("generate_transform", phase4.generate_transform)
    builder.add_node("verify_crr", phase5.verify_crr)
    builder.add_node("repair_crr", phase5.repair_crr)
    builder.add_node("finalize_verified", phase5.finalize_verified)
    builder.add_node("finalize_with_issues", phase5.finalize_with_issues)
    builder.add_node("finalize_index_only", phase3.finalize_index_only)
    builder.add_node("finalize_error", phase3.finalize_error)

    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        _route_after_initialize,
        {"ingest": "ingest_sources", "classify": "classify_intent", "error": "finalize_error"},
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
    builder.add_conditional_edges(
        "build_context",
        _route_generation,
        {"qa": "generate_qa", "transform": "generate_transform", "error": "finalize_error"},
    )
    builder.add_conditional_edges(
        "generate_qa", _route_after_generation, {"verify": "verify_crr", "error": "finalize_error"}
    )
    builder.add_conditional_edges(
        "generate_transform", _route_after_generation, {"verify": "verify_crr", "error": "finalize_error"}
    )
    builder.add_conditional_edges(
        "verify_crr",
        _route_verification,
        {
            "pass": "finalize_verified",
            "repair": "repair_crr",
            "issues": "finalize_with_issues",
            "error": "finalize_error",
        },
    )
    builder.add_conditional_edges(
        "repair_crr",
        _route_after_generation,
        {"verify": "verify_crr", "error": "finalize_error"},
    )
    builder.add_edge("finalize_verified", END)
    builder.add_edge("finalize_with_issues", END)
    builder.add_edge("finalize_index_only", END)
    builder.add_edge("finalize_error", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver(), name="phase5-orchestrator")


class Phase5Orchestrator(Phase4Orchestrator):
    def __init__(
        self,
        graph,
        *,
        default_top_k: int = 5,
        session_manager: UserSessionManager | None = None,
        pipeline: Phase12Pipeline | None = None,
        max_repair_attempts: int = 2,
    ):
        super().__init__(
            graph,
            default_top_k=default_top_k,
            session_manager=session_manager,
            pipeline=pipeline,
        )
        self.max_repair_attempts = max(0, int(max_repair_attempts))

    def invoke(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        source_paths: list[str] | None = None,
        query: str = "",
        request_mode: str = "auto",
        selected_source_ids: list[str] | None = None,
        top_k: int | None = None,
        session_title: str | None = None,
        transformation_config: TransformationConfig | dict[str, Any] | None = None,
        retrieval_config: dict[str, Any] | None = None,
    ) -> AgentState:
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("user_id is required")
        if session_id is None:
            if not self.session_manager:
                raise ValueError("session_id is required when no session manager is configured")
            session_id = self.session_manager.create(user_id, title=session_title).session_id
        elif self.session_manager:
            self.session_manager.ensure(user_id, session_id, title=session_title)

        config_obj = (
            transformation_config
            if isinstance(transformation_config, TransformationConfig)
            else TransformationConfig.model_validate(transformation_config or {})
        )
        payload: AgentState = {
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "request_mode": request_mode,
            "pending_source_paths": list(source_paths or []),
            "selected_source_ids": list(selected_source_ids or []),
            "top_k": int(top_k or self.default_top_k),
            "transformation_config": config_obj.model_dump(mode="json"),
            "retrieval_config": dict(retrieval_config or {}),
            "retrieved_documents": [],
            "context_groups": [],
            "prepared_context": "",
            "canonical_response": {},
            "section_digests": [],
            "generation_metadata": {},
            "verification_report": {},
            "verification_metadata": {},
            "verification_requires_repair": False,
            "repair_attempts": 0,
            "max_repair_attempts": self.max_repair_attempts,
            "artifacts": [],
            "artifact_metadata": {},
            "warnings": [],
            "errors": [],
        }
        graph_config = {"configurable": {"thread_id": make_thread_id(user_id, session_id)}}
        return self.graph.invoke(payload, config=graph_config)
