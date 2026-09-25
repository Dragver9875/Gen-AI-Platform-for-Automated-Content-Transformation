from __future__ import annotations
import json
import statistics
import time
from pathlib import Path
from typing import Any

from evaluation.artifacts import artifact_metrics
from evaluation.grounding import grounding_metrics
from evaluation.models import EvalCase, EvalCaseResult, EvalDataset, EvalReport
from evaluation.retrieval import retrieval_metrics
from evaluation.runtime import runtime_metrics
from evaluation.telemetry import telemetry_session
from evaluation.text import reference_metrics


def _chunk_id(doc: dict[str, Any]) -> str:
    meta = dict(doc.get("metadata") or {})
    return str(meta.get("chunk_id") or doc.get("id") or "")


def _flatten_crr(state: dict[str, Any]) -> str:
    # Prefer the user-facing text artifact when available; otherwise evaluate the
    # format-independent CRR so reference metrics still work for PDF/PPTX-only cases.
    for artifact in state.get("artifacts") or []:
        if str(artifact.get("format") or "").lower() == "text" and artifact.get("status") == "generated" and artifact.get("path"):
            path = Path(str(artifact["path"]))
            if path.exists() and path.is_file():
                try:
                    return path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    pass
    crr = dict(state.get("canonical_response") or {})
    pieces = [str(crr.get("title") or ""), str(crr.get("summary") or "")]
    for sec in crr.get("sections") or []:
        pieces.extend([str(sec.get("heading") or ""), str(sec.get("content") or "")])
        pieces.extend(str(x) for x in (sec.get("bullets") or []))
    return "\n".join(x for x in pieces if x.strip())


def _numeric_leaves(tree: dict[str, Any], prefix: str = ""):
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _numeric_leaves(value, name)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield name, float(value)


class EvaluationRunner:
    def __init__(self, agent, *, pricing: dict[str, Any] | None = None):
        self.agent = agent
        self.pricing = pricing or {}

    def run_case(self, case: EvalCase) -> EvalCaseResult:
        started = time.perf_counter()
        try:
            with telemetry_session() as collector:
                state = self.agent.invoke(
                    user_id=case.user_id,
                    session_id=case.session_id,
                    source_paths=case.source_paths,
                    query=case.query,
                    request_mode=case.request_mode,
                    selected_source_ids=case.selected_source_ids,
                    top_k=case.top_k,
                    transformation_config=case.transformation_config,
                    retrieval_config=case.retrieval_config,
                )
            wall_ms = (time.perf_counter() - started) * 1000.0
            telemetry = collector.summary(self.pricing)
            metrics: dict[str, Any] = {
                "grounding": grounding_metrics(state.get("verification_report")),
                "artifacts": artifact_metrics(state.get("artifacts"), case.expected_formats or case.transformation_config.get("output_formats", [])),
                "runtime": runtime_metrics(telemetry, wall_ms),
            }
            if case.relevant_chunk_ids:
                ranked = [_chunk_id(d) for d in (state.get("retrieved_documents") or []) if _chunk_id(d)]
                metrics["retrieval"] = retrieval_metrics(ranked, case.relevant_chunk_ids, case.top_k or len(ranked) or 1)
            if case.reference_text is not None:
                metrics["reference"] = reference_metrics(_flatten_crr(state), case.reference_text)
            status = str(state.get("status") or "unknown")
            passed = status in {"phase6_complete", "phase5_complete"} and not state.get("errors")
            return EvalCaseResult(
                case_id=case.case_id,
                status=status,
                passed=passed,
                metrics=metrics,
                warnings=list(state.get("warnings") or []),
                errors=list(state.get("errors") or []),
                telemetry=telemetry,
            )
        except Exception as exc:
            return EvalCaseResult(case_id=case.case_id, status="exception", passed=False, errors=[str(exc)], metrics={"runtime": {"end_to_end_latency_ms": (time.perf_counter()-started)*1000.0}})

    def run(self, dataset: EvalDataset) -> EvalReport:
        results = [self.run_case(case) for case in dataset.cases]
        buckets: dict[str, list[float]] = {}
        for result in results:
            for name, value in _numeric_leaves(result.metrics):
                buckets.setdefault(name, []).append(value)
        aggregate = {
            "case_count": len(results),
            "pass_rate": sum(r.passed for r in results) / len(results) if results else 0.0,
            "metrics": {name: statistics.fmean(values) for name, values in sorted(buckets.items()) if values},
        }
        return EvalReport(dataset_name=dataset.name, dataset_version=dataset.version, cases=results, aggregate=aggregate)

    @staticmethod
    def load_dataset(path: str | Path) -> EvalDataset:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            cases = [EvalCase.model_validate(json.loads(line)) for line in text.splitlines() if line.strip()]
            return EvalDataset(name=path.stem, cases=cases)
        data = json.loads(text)
        if isinstance(data, list):
            return EvalDataset(name=path.stem, cases=[EvalCase.model_validate(x) for x in data])
        return EvalDataset.model_validate(data)

    @staticmethod
    def save_report(report: EvalReport, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return path
