from __future__ import annotations
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

from app.config import Settings
from app.factory import build_phase6
from evaluation.runner import EvaluationRunner
from evaluation.reporting import save_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 9 evaluation suite against the Phase 6 pipeline")
    parser.add_argument("--dataset", required=True, help="JSON or JSONL evaluation dataset")
    parser.add_argument("--output", default="eval_reports/latest.json")
    parser.add_argument("--pricing-json", help="Optional JSON file mapping telemetry provider names to pricing")
    args = parser.parse_args()

    load_dotenv()
    pricing = json.loads(Path(args.pricing_json).read_text()) if args.pricing_json else {}
    agent = build_phase6(Settings.from_env())
    runner = EvaluationRunner(agent, pricing=pricing)
    dataset = runner.load_dataset(args.dataset)
    report = runner.run(dataset)
    path = runner.save_report(report, args.output)
    markdown_path = save_markdown(report, Path(args.output).with_suffix(".md"))
    print(json.dumps({"report": str(path), "markdown_report": str(markdown_path), "aggregate": report.aggregate}, indent=2))


if __name__ == "__main__":
    main()
