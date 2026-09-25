from __future__ import annotations

import argparse
import json

from app.config import Settings
from app.factory import build_phase5


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 verified content generation")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--file", action="append", dest="files", default=[])
    parser.add_argument("--query", default="")
    parser.add_argument("--mode", choices=["auto", "qa", "transform"], default="auto")
    parser.add_argument("--artifact-type", default="auto")
    parser.add_argument("--tone", default="professional")
    parser.add_argument("--audience", default="general")
    parser.add_argument("--language", default="English")
    parser.add_argument("--detail", default="medium")
    parser.add_argument("--objective", default="inform")
    parser.add_argument("--style", default="clear")
    args = parser.parse_args()

    agent = build_phase5(Settings.from_env())
    state = agent.invoke(
        user_id=args.user_id,
        session_id=args.session_id,
        source_paths=args.files,
        query=args.query,
        request_mode=args.mode,
        transformation_config={
            "artifact_type": args.artifact_type,
            "tone": args.tone,
            "audience": args.audience,
            "language": args.language,
            "detail_level": args.detail,
            "objective": args.objective,
            "style": args.style,
        },
    )
    print(json.dumps({
        "status": state.get("status"),
        "session_id": state.get("session_id"),
        "intent": state.get("detected_intent"),
        "generation_metadata": state.get("generation_metadata"),
        "verification_metadata": state.get("verification_metadata"),
        "verification_report": state.get("verification_report"),
        "canonical_response": state.get("canonical_response"),
        "repair_attempts": state.get("repair_attempts"),
        "warnings": state.get("warnings"),
        "errors": state.get("errors"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
