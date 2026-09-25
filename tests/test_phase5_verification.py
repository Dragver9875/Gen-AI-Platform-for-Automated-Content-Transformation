from __future__ import annotations

from pathlib import Path

from agents.phase5_graph import Phase5Orchestrator, build_phase5_graph
from app.schemas import Chunk, IngestionResult, SourceElement
from app.sessions import InMemorySessionStore, UserSessionManager
from generation.service import GenerationService
from verification.literals import DeterministicLiteralValidator
from verification.service import VerificationService


class FakeRetriever:
    def __init__(self):
        self.docs = [
            {
                "id": "c1",
                "text": "Threat activity increased by 32 percent during Q3.",
                "metadata": {
                    "chunk_id": "c1",
                    "user_id": "u1",
                    "session_id": "s1",
                    "source_id": "src-1",
                    "filename": "report.txt",
                    "section": "Overview",
                    "page_start": 1,
                    "page_end": 1,
                    "slide_start": -1,
                    "slide_end": -1,
                },
            },
            {
                "id": "c2",
                "text": "Enable multi-factor authentication for privileged accounts.",
                "metadata": {
                    "chunk_id": "c2",
                    "user_id": "u1",
                    "session_id": "s1",
                    "source_id": "src-1",
                    "filename": "report.txt",
                    "section": "Recommendations",
                    "page_start": 2,
                    "page_end": 2,
                    "slide_start": -1,
                    "slide_end": -1,
                },
            },
        ]

    def retrieve(self, query, *, where=None, final_k=5):
        return self.docs[:final_k]

    def get_corpus(self, *, where=None, limit=None):
        return self.docs[:limit] if limit else list(self.docs)


class FakePipeline:
    def __init__(self):
        self.retriever = FakeRetriever()

    def ingest_and_index(self, path, *, user_id, session_id):
        result = IngestionResult(
            source_id="src-1",
            filename=Path(path).name,
            media_type="text/plain",
            strategy="text-direct",
            elements=[SourceElement("e1", "paragraph", "source")],
        )
        chunk = Chunk(
            "c1",
            "source",
            {
                "chunk_id": "c1",
                "user_id": user_id,
                "session_id": session_id,
                "source_id": "src-1",
                "filename": Path(path).name,
            },
        )
        return result, [chunk]


class GenerationLLM:
    def __init__(self, *, wrong_number: bool = False, missing_evidence: bool = False):
        self.wrong_number = wrong_number
        self.missing_evidence = missing_evidence

    def generate_json(self, *, system_prompt, user_prompt, schema_name, json_schema, temperature, max_tokens):
        if schema_name == "section_digest":
            return {
                "source_id": "src-1",
                "section": "Overview",
                "summary": "Digest",
                "key_points": [],
                "claims": [],
            }
        value = "42" if self.wrong_number else "32"
        evidence = [] if self.missing_evidence else ["c1"]
        return {
            "crr_version": "1.0",
            "artifact_type": "answer",
            "title": "Threat Brief",
            "summary": f"Threat activity increased by {value} percent.",
            "sections": [{"heading": "Answer", "content": "Grounded answer", "bullets": []}],
            "key_points": [],
            "claims": [
                {
                    "claim_id": "claim_1",
                    "text": f"Threat activity increased by {value} percent.",
                    "evidence": evidence,
                }
            ],
            "rendering": {
                "tone": "professional",
                "audience": "general",
                "language": "English",
                "detail_level": "medium",
                "objective": "inform",
                "style": "clear",
            },
            "insufficiencies": [],
        }


class SemanticVerifierLLM:
    """Deliberately optimistic; deterministic checks must still catch conflicts."""

    def __init__(self, *, always_unsupported: bool = False):
        self.calls = 0
        self.always_unsupported = always_unsupported

    def generate_json(self, *, system_prompt, user_prompt, schema_name, json_schema, temperature, max_tokens):
        self.calls += 1
        if schema_name != "semantic_verification_batch":
            raise AssertionError(schema_name)
        status = "unsupported" if self.always_unsupported else "supported"
        return {
            "claims": [
                {
                    "claim_id": "claim_1",
                    "status": status,
                    "confidence": 0.99,
                    "supported_evidence": ["c1"] if status == "supported" else [],
                    "rationale": "Test semantic assessment.",
                    "unsupported_fragments": [] if status == "supported" else ["claim"],
                    "suggested_correction": "Use the source value." if status != "supported" else "",
                }
            ]
        }


class RepairLLM:
    def __init__(self, *, fix: bool = True):
        self.calls = 0
        self.fix = fix

    def generate_json(self, *, system_prompt, user_prompt, schema_name, json_schema, temperature, max_tokens):
        self.calls += 1
        value = "32" if self.fix else "42"
        return {
            "crr_version": "1.0",
            "artifact_type": "answer",
            "title": "Threat Brief",
            "summary": f"Threat activity increased by {value} percent.",
            "sections": [{"heading": "Answer", "content": "Repaired answer", "bullets": []}],
            "key_points": [],
            "claims": [
                {
                    "claim_id": "claim_1",
                    "text": f"Threat activity increased by {value} percent.",
                    "evidence": ["c1"],
                }
            ],
            "rendering": {
                "tone": "professional",
                "audience": "general",
                "language": "English",
                "detail_level": "medium",
                "objective": "inform",
                "style": "clear",
            },
            "insufficiencies": [],
        }


def make_agent(*, wrong_number=False, missing_evidence=False, repair_fix=True, max_repairs=2, verifier_always_unsupported=False):
    pipeline = FakePipeline()
    generation_llm = GenerationLLM(wrong_number=wrong_number, missing_evidence=missing_evidence)
    generator = GenerationService(generation_llm)
    semantic = SemanticVerifierLLM(always_unsupported=verifier_always_unsupported)
    repair = RepairLLM(fix=repair_fix)
    verifier = VerificationService(
        semantic,
        repair_generator=repair,
        max_repair_attempts=max_repairs,
        min_faithfulness_score=1.0,
    )
    graph = build_phase5_graph(pipeline, generator, verifier, default_top_k=5, context_max_chars=10000)
    sessions = UserSessionManager(InMemorySessionStore())
    return Phase5Orchestrator(
        graph,
        session_manager=sessions,
        pipeline=pipeline,
        max_repair_attempts=max_repairs,
    ), semantic, repair


def run(agent, tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("source", encoding="utf-8")
    return agent.invoke(
        user_id="u1",
        session_id="s1",
        source_paths=[str(source)],
        query="What changed?",
        request_mode="qa",
    )


def test_supported_claim_passes_without_repair(tmp_path):
    agent, semantic, repair = make_agent()
    state = run(agent, tmp_path)
    assert state["status"] == "phase5_complete"
    assert state["verification_report"]["passed"] is True
    assert state["verification_report"]["faithfulness_score"] == 1.0
    assert state["repair_attempts"] == 0
    assert semantic.calls == 1
    assert repair.calls == 0


def test_numeric_conflict_overrides_semantic_support_and_is_repaired(tmp_path):
    agent, semantic, repair = make_agent(wrong_number=True)
    state = run(agent, tmp_path)
    assert state["status"] == "phase5_complete"
    assert state["repair_attempts"] == 1
    assert "32 percent" in state["canonical_response"]["claims"][0]["text"]
    assert state["verification_report"]["passed"] is True
    assert semantic.calls == 2
    assert repair.calls == 1


def test_missing_cited_evidence_routes_to_repair_limit(tmp_path):
    agent, _, repair = make_agent(missing_evidence=True, repair_fix=False, max_repairs=1)
    state = run(agent, tmp_path)
    # Repair adds cited evidence, but intentionally retains the wrong value; the
    # deterministic conflict remains after the one allowed repair.
    assert state["status"] == "phase5_complete_with_issues"
    assert state["repair_attempts"] == 1
    assert state["verification_report"]["passed"] is False
    assert repair.calls == 1


def test_semantic_unsupported_exhausts_bounded_repair(tmp_path):
    agent, semantic, repair = make_agent(
        verifier_always_unsupported=True,
        repair_fix=True,
        max_repairs=2,
    )
    state = run(agent, tmp_path)
    assert state["status"] == "phase5_complete_with_issues"
    assert state["repair_attempts"] == 2
    assert semantic.calls == 3
    assert repair.calls == 2


def test_deterministic_literal_validator_flags_clear_percentage_conflict():
    issues = DeterministicLiteralValidator().validate(
        "Incidents increased by 42%.",
        "The report says incidents increased by 32%.",
    )
    assert any(issue.kind == "numeric_conflict" and issue.severity == "conflict" for issue in issues)
