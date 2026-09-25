from __future__ import annotations

from agents.phase6_graph import Phase6Orchestrator, build_phase6_graph
from app.phase12 import Phase12Pipeline
from app.sessions import UserSessionManager
from artifacts.service import ArtifactService
from generation.service import GenerationService
from verification.service import VerificationService


def create_phase6_orchestrator(
    pipeline: Phase12Pipeline,
    generator: GenerationService,
    verifier: VerificationService,
    artifacts: ArtifactService,
    *,
    default_top_k: int = 5,
    context_max_chars: int = 60000,
    checkpointer=None,
    session_manager: UserSessionManager | None = None,
) -> Phase6Orchestrator:
    graph = build_phase6_graph(
        pipeline, generator, verifier, artifacts,
        default_top_k=default_top_k, context_max_chars=context_max_chars, checkpointer=checkpointer,
    )
    return Phase6Orchestrator(
        graph, default_top_k=default_top_k, session_manager=session_manager, pipeline=pipeline,
        max_repair_attempts=verifier.max_repair_attempts,
    )
