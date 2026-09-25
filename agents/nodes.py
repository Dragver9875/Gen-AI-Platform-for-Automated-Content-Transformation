from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.context import group_hierarchical, group_qa, render_context
from agents.intent import IntentRouter
from agents.state import AgentState
from app.phase12 import Phase12Pipeline


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def _where(user_id: str, session_id: str, source_ids: list[str] | None = None) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = [{"user_id": user_id}, {"session_id": session_id}]
    ids = _unique(source_ids or [])
    if len(ids) == 1:
        clauses.append({"source_id": ids[0]})
    elif len(ids) > 1:
        clauses.append({"source_id": {"$in": ids}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class Phase3Nodes:
    def __init__(
        self,
        pipeline: Phase12Pipeline,
        *,
        default_top_k: int = 5,
        context_max_chars: int = 60000,
        intent_router: IntentRouter | None = None,
    ):
        self.pipeline = pipeline
        self.default_top_k = default_top_k
        self.context_max_chars = context_max_chars
        self.intent_router = intent_router or IntentRouter()

    def initialize(self, state: AgentState) -> dict[str, Any]:
        user_id = str(state.get("user_id") or "").strip()
        session_id = str(state.get("session_id") or "").strip()
        if not user_id:
            return {
                "status": "error",
                "errors": ["user_id is required."],
                "detected_intent": "error",
            }
        if not session_id:
            return {
                "status": "error",
                "errors": ["session_id is required."],
                "detected_intent": "error",
            }
        return {
            "user_id": user_id,
            "session_id": session_id,
            "status": "initialized",
            "warnings": list(state.get("warnings") or []),
            "errors": list(state.get("errors") or []),
            "active_source_ids": list(state.get("active_source_ids") or []),
            "ingested_sources": list(state.get("ingested_sources") or []),
            "top_k": int(state.get("top_k") or self.default_top_k),
        }

    def ingest_sources(self, state: AgentState) -> dict[str, Any]:
        pending = [str(path) for path in (state.get("pending_source_paths") or []) if str(path).strip()]
        active = list(state.get("active_source_ids") or [])
        ingested = list(state.get("ingested_sources") or [])
        warnings = list(state.get("warnings") or [])
        errors = list(state.get("errors") or [])

        for raw_path in pending:
            path = Path(raw_path)
            if not path.exists():
                errors.append(f"Source file not found: {raw_path}")
                continue
            try:
                result, chunks = self.pipeline.ingest_and_index(path, user_id=state["user_id"], session_id=state["session_id"])
                active.append(result.source_id)
                warnings.extend(result.warnings)
                ingested.append({
                    "source_id": result.source_id,
                    "filename": result.filename,
                    "media_type": result.media_type,
                    "strategy": result.strategy,
                    "elements": len(result.elements),
                    "chunks": len(chunks),
                    "warnings": result.warnings,
                })
            except Exception as exc:
                errors.append(f"Failed to ingest {raw_path}: {exc}")

        return {
            "pending_source_paths": [],
            "active_source_ids": _unique(active),
            "ingested_sources": ingested,
            "warnings": _unique(warnings),
            "errors": errors,
            "status": "indexed" if not errors else "indexed_with_errors",
        }

    def classify_intent(self, state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors") or [])
        query = str(state.get("query") or "").strip()
        mode = state.get("request_mode") or "auto"
        decision = self.intent_router.classify(query, mode)

        if errors and not (state.get("active_source_ids") or []):
            return {
                "detected_intent": "error",
                "intent_confidence": 1.0,
                "intent_reason": "Ingestion failed and no indexed source is available.",
                "status": "error",
                "errors": errors,
            }

        if decision.intent in {"qa", "transform"} and not (state.get("active_source_ids") or []):
            errors.append("No indexed sources are active for this session.")
            return {
                "detected_intent": "error",
                "intent_confidence": 1.0,
                "intent_reason": "A retrieval request requires at least one indexed source.",
                "status": "error",
                "errors": errors,
            }

        return {
            "detected_intent": decision.intent,
            "intent_confidence": decision.confidence,
            "intent_reason": decision.reason,
            "status": "routed",
            "errors": errors,
        }

    def retrieve_qa(self, state: AgentState) -> dict[str, Any]:
        selected = list(state.get("selected_source_ids") or state.get("active_source_ids") or [])
        where = _where(state["user_id"], state["session_id"], selected)
        retrieval_kwargs: dict[str, Any] = {
            "where": where,
            "final_k": int(state.get("top_k") or self.default_top_k),
        }
        if state.get("retrieval_config"):
            retrieval_kwargs["config"] = state.get("retrieval_config")
        docs = self.pipeline.retriever.retrieve(str(state.get("query") or ""), **retrieval_kwargs)
        return {
            "retrieved_documents": docs,
            "retrieval_mode": "hybrid_top_k",
            "status": "retrieved",
        }

    def retrieve_transform(self, state: AgentState) -> dict[str, Any]:
        selected = list(state.get("selected_source_ids") or state.get("active_source_ids") or [])
        where = _where(state["user_id"], state["session_id"], selected)
        docs = self.pipeline.retriever.get_corpus(where=where)
        return {
            "retrieved_documents": docs,
            "retrieval_mode": "hierarchical_full_document",
            "status": "retrieved",
        }

    def build_context(self, state: AgentState) -> dict[str, Any]:
        docs = list(state.get("retrieved_documents") or [])
        intent = state.get("detected_intent")
        groups = group_hierarchical(docs) if intent == "transform" else group_qa(docs)
        prepared = render_context(groups, max_chars=self.context_max_chars)
        warnings = list(state.get("warnings") or [])
        if not docs:
            warnings.append("Retrieval returned no matching chunks.")
        return {
            "context_groups": groups,
            "prepared_context": prepared,
            "warnings": _unique(warnings),
            "status": "phase4_ready" if docs else "phase4_ready_no_results",
        }

    def finalize_index_only(self, state: AgentState) -> dict[str, Any]:
        return {
            "retrieved_documents": [],
            "context_groups": [],
            "prepared_context": "",
            "retrieval_mode": "none",
            "status": "indexed_ready",
        }

    def finalize_error(self, state: AgentState) -> dict[str, Any]:
        return {"status": "error"}
