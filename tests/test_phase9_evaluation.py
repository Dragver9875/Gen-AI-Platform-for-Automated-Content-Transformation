from pathlib import Path

from evaluation.artifacts import artifact_metrics
from evaluation.grounding import grounding_metrics
from evaluation.models import EvalCase, EvalDataset
from evaluation.retrieval import retrieval_metrics
from evaluation.runner import EvaluationRunner
from evaluation.telemetry import TelemetryEvent, telemetry_session, record_event, record_usage
from evaluation.text import reference_metrics


def test_retrieval_metrics_known_ranking():
    m = retrieval_metrics(["a", "x", "b", "z"], ["a", "b"], 3)
    assert round(m["precision@3"], 4) == round(2/3, 4)
    assert m["recall@3"] == 1.0
    assert m["hit_rate@3"] == 1.0
    assert m["mrr"] == 1.0
    assert 0.0 < m["ndcg@3"] <= 1.0


def test_reference_metrics_identical_text():
    m = reference_metrics("Alpha beta gamma", "Alpha beta gamma")
    assert m["exact_match"] == 1.0
    assert m["token_f1"] == 1.0
    assert m["rouge_l_f1"] == 1.0
    assert m["length_ratio"] == 1.0


def test_grounding_metrics_from_phase5_report():
    report = {
        "faithfulness_score": 0.75,
        "repair_attempts": 1,
        "claims": [
            {"status": "supported", "confidence": 0.9},
            {"status": "partially_supported", "confidence": 0.7},
        ],
    }
    m = grounding_metrics(report)
    assert m["faithfulness"] == 0.75
    assert m["claim_support_rate"] == 0.5
    assert m["claim_partial_rate"] == 0.5
    assert m["repair_attempts"] == 1.0


def test_artifact_metrics(tmp_path):
    output = tmp_path / "answer.txt"
    output.write_text("hello", encoding="utf-8")
    m = artifact_metrics([
        {"format": "text", "status": "generated", "path": str(output)},
        {"format": "pdf", "status": "failed", "path": None},
    ], ["text", "pdf"])
    assert m["artifact_success_rate"] == 0.5
    assert m["artifact_failure_rate"] == 0.5
    assert m["artifact_nonempty_file_rate"] == 0.5
    assert m["expected_format_coverage"] == 0.5


def test_telemetry_summary_with_pricebook():
    with telemetry_session() as collector:
        record_event(TelemetryEvent("llm", "POST", 100.0, True, 2))
        record_usage("llm", "generation", input_tokens=1000, output_tokens=500)
    summary = collector.summary({"llm": {"input_per_million": 2.0, "output_per_million": 4.0}})
    assert summary["provider_request_count"] == 1
    assert summary["provider_retry_count"] == 1
    assert summary["input_tokens"] == 1000
    assert summary["output_tokens"] == 500
    assert round(summary["estimated_cost"], 6) == 0.004


class FakeAgent:
    def __init__(self, artifact_path: str):
        self.artifact_path = artifact_path

    def invoke(self, **kwargs):
        record_event(TelemetryEvent("harrier", "POST", 20.0, True, 1))
        return {
            "status": "phase6_complete",
            "retrieved_documents": [
                {"id": "a", "metadata": {"chunk_id": "a"}},
                {"id": "x", "metadata": {"chunk_id": "x"}},
            ],
            "canonical_response": {
                "title": "Answer", "summary": "Alpha beta", "sections": [], "claims": []
            },
            "verification_report": {
                "faithfulness_score": 1.0,
                "repair_attempts": 0,
                "claims": [{"status": "supported", "confidence": 0.95}],
            },
            "artifacts": [{"format": "text", "status": "generated", "path": self.artifact_path}],
            "warnings": [], "errors": [],
        }


def test_evaluation_runner_aggregates(tmp_path):
    artifact = tmp_path / "answer.txt"
    artifact.write_text("Alpha beta", encoding="utf-8")
    runner = EvaluationRunner(FakeAgent(str(artifact)))
    dataset = EvalDataset(name="unit", cases=[EvalCase(
        case_id="c1", query="q", relevant_chunk_ids=["a"], reference_text="Alpha beta",
        expected_formats=["text"], transformation_config={"output_formats": ["text"]}, top_k=2,
    )])
    report = runner.run(dataset)
    assert report.aggregate["case_count"] == 1
    assert report.aggregate["pass_rate"] == 1.0
    assert report.cases[0].metrics["retrieval"]["recall@2"] == 1.0
    assert report.cases[0].metrics["reference"]["token_f1"] == 1.0
    assert report.cases[0].metrics["grounding"]["faithfulness"] == 1.0
