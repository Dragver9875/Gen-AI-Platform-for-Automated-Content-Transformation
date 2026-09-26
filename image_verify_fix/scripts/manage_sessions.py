from __future__ import annotations

import argparse
import json

from app.config import Settings
from app.factory import build_phase3


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/list/rename/delete user sessions.")
    parser.add_argument("--user-id", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create")
    create.add_argument("--session-id")
    create.add_argument("--title")

    listing = sub.add_parser("list")
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--offset", type=int, default=0)

    rename = sub.add_parser("rename")
    rename.add_argument("session_id")
    rename.add_argument("title")

    delete = sub.add_parser("delete")
    delete.add_argument("session_id")
    delete.add_argument("--keep-vectors", action="store_true")

    args = parser.parse_args()
    agent = build_phase3(Settings.from_env())

    if args.command == "create":
        result = agent.create_session(args.user_id, session_id=args.session_id, title=args.title)
    elif args.command == "list":
        result = agent.list_sessions(args.user_id, limit=args.limit, offset=args.offset)
    elif args.command == "rename":
        result = agent.rename_session(args.user_id, args.session_id, args.title)
    else:
        agent.delete_session(args.user_id, args.session_id, delete_vectors=not args.keep_vectors)
        result = {"deleted": True, "session_id": args.session_id}

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
