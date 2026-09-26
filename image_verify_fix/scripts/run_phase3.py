from __future__ import annotations

import argparse
import json

from app.config import Settings
from app.factory import build_phase3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 3 LangGraph orchestrator.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--file", action="append", default=[], help="Source file; repeat for multiple files.")
    parser.add_argument("--query", default="")
    parser.add_argument("--mode", choices=["auto", "qa", "transform"], default="auto")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    agent = build_phase3(Settings.from_env())
    state = agent.invoke(
        user_id=args.user_id,
        session_id=args.session_id,
        query=args.query,
        source_paths=args.file,
        request_mode=args.mode,
        top_k=args.top_k,
    )
    summary = {
        "status": state.get("status"),
        "intent": state.get("detected_intent"),
        "intent_reason": state.get("intent_reason"),
        "active_source_ids": state.get("active_source_ids", []),
        "ingested_sources": state.get("ingested_sources", []),
        "retrieval_mode": state.get("retrieval_mode"),
        "retrieved_documents": len(state.get("retrieved_documents", [])),
        "context_groups": len(state.get("context_groups", [])),
        "prepared_context_chars": len(state.get("prepared_context", "")),
        "warnings": state.get("warnings", []),
        "errors": state.get("errors", []),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
