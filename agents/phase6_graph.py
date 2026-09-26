from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.graph import _route_after_initialize, _route_intent
from agents.nodes import Phase3Nodes
from agents.phase4_graph import _route_generation
from agents.phase4_nodes import Phase4Nodes
from agents.phase5_graph import Phase5Orchestrator, _route_after_generation, _route_after_repair, _route_verification
from agents.phase5_nodes import Phase5Nodes
from agents.phase6_nodes import Phase6Nodes
from agents.state import AgentState
from app.phase12 import Phase12Pipeline
from artifacts.service import ArtifactService
from generation.service import GenerationService
from verification.service import VerificationService


def _route_after_artifacts(state: AgentState) -> str:
    return "error" if state.get("status") == "error" else "done"


def build_phase6_graph(
    pipeline: Phase12Pipeline,
    generator: GenerationService,
    verifier: VerificationService,
    artifacts: ArtifactService,
    *,
    default_top_k: int = 5,
    context_max_chars: int = 60000,
    checkpointer: Any | None = None,
):
    # Deliberately preserve the Phase 5 topology. Phase 6 appends one generic
    # artifact node after a successful verification; it does not create format-
    # specific graph branches.
    phase3 = Phase3Nodes(pipeline, default_top_k=default_top_k, context_max_chars=context_max_chars)
    phase4 = Phase4Nodes(generator)
    phase5 = Phase5Nodes(verifier)
    phase6 = Phase6Nodes(artifacts)

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
    builder.add_node("generate_artifacts", phase6.generate_artifacts)
    builder.add_node("finalize_index_only", phase3.finalize_index_only)
    builder.add_node("finalize_error", phase3.finalize_error)

    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", _route_after_initialize, {"ingest": "ingest_sources", "classify": "classify_intent", "error": "finalize_error"})
    builder.add_edge("ingest_sources", "classify_intent")
    builder.add_conditional_edges("classify_intent", _route_intent, {"qa": "retrieve_qa", "transform": "retrieve_transform", "index_only": "finalize_index_only", "error": "finalize_error"})
    builder.add_edge("retrieve_qa", "build_context")
    builder.add_edge("retrieve_transform", "build_context")
    builder.add_conditional_edges("build_context", _route_generation, {"qa": "generate_qa", "transform": "generate_transform", "error": "finalize_error"})
    builder.add_conditional_edges("generate_qa", _route_after_generation, {"verify": "verify_crr", "error": "finalize_error"})
    builder.add_conditional_edges("generate_transform", _route_after_generation, {"verify": "verify_crr", "error": "finalize_error"})
    builder.add_conditional_edges("verify_crr", _route_verification, {"pass": "finalize_verified", "repair": "repair_crr", "issues": "finalize_with_issues", "error": "finalize_error"})
    builder.add_conditional_edges("repair_crr", _route_after_repair, {"verify": "verify_crr", "issues": "finalize_with_issues", "error": "finalize_error"})
    builder.add_edge("finalize_verified", "generate_artifacts")
    builder.add_conditional_edges("generate_artifacts", _route_after_artifacts, {"done": END, "error": "finalize_error"})
    # Unresolved factual issues intentionally do not render final artifacts.
    builder.add_edge("finalize_with_issues", END)
    builder.add_edge("finalize_index_only", END)
    builder.add_edge("finalize_error", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver(), name="phase6-orchestrator")


class Phase6Orchestrator(Phase5Orchestrator):
    def invoke_task(self, task):
        from core.task_spec import TaskSpec
        spec = task if isinstance(task, TaskSpec) else TaskSpec.model_validate(task)
        return self.invoke(
            user_id=spec.user_id,
            session_id=spec.session_id,
            source_paths=spec.source_paths,
            query=spec.query,
            request_mode=spec.request_mode,
            selected_source_ids=spec.selected_source_ids,
            session_title=spec.session_title,
            transformation_config=spec.transformation,
            retrieval_config=spec.retrieval.model_dump(mode="json") if spec.retrieval else None,
        )
