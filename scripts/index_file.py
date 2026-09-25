from __future__ import annotations

import argparse
import json

from app.config import Settings
from app.factory import build_phase12


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, chunk, index and optionally retrieve from a file.")
    parser.add_argument("file")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = Settings.from_env()
    pipeline = build_phase12(settings)
    result, chunks = pipeline.ingest_and_index(args.file, session_id=args.session_id)

    print(json.dumps({
        "source_id": result.source_id,
        "strategy": result.strategy,
        "elements": len(result.elements),
        "chunks": len(chunks),
        "warnings": result.warnings,
    }, indent=2))

    if args.query:
        docs = pipeline.retrieve(args.query, session_id=args.session_id, source_id=result.source_id, final_k=args.top_k)
        print(json.dumps(docs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
