from __future__ import annotations

import argparse
import json
from dotenv import load_dotenv

from app.config import Settings
from app.factory import build_phase6
from scripts.query_transport import decode_query_b64


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the content transformation pipeline through Phase 6 artifact generation")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--file", action="append", dest="files", default=[])
    parser.add_argument("--query", default="")
    parser.add_argument("--query-b64", default="", help="UTF-8 Base64 encoded query; preferred for Windows PowerShell native-process calls")
    parser.add_argument("--mode", choices=["auto", "qa", "transform"], default="auto")
    parser.add_argument("--artifact-type", default="auto")
    parser.add_argument("--format", action="append", dest="formats", default=[])
    parser.add_argument("--audience", default="general")
    parser.add_argument("--tone", default="professional")
    parser.add_argument("--language", default="English")
    parser.add_argument("--detail", default="medium")
    parser.add_argument("--verification-profile", default="strict")
    args = parser.parse_args()

    query = args.query
    if args.query_b64:
        try:
            query = decode_query_b64(args.query_b64)
        except ValueError as exc:
            parser.error(str(exc))

    load_dotenv()
    settings = Settings.from_env()
    agent = build_phase6(settings)
    state = agent.invoke(
        user_id=args.user_id,
        session_id=args.session_id,
        source_paths=args.files,
        query=query,
        request_mode=args.mode,
        transformation_config={
            "artifact_type": args.artifact_type,
            "output_formats": args.formats or ["text"],
            "audience": args.audience,
            "tone": args.tone,
            "language": args.language,
            "detail_level": args.detail,
            "verification_profile": args.verification_profile,
        },
    )
    print(json.dumps({
        "status": state.get("status"),
        "session_id": state.get("session_id"),
        "verification": state.get("verification_report"),
        "artifacts": state.get("artifacts"),
        "warnings": state.get("warnings"),
        "errors": state.get("errors"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
