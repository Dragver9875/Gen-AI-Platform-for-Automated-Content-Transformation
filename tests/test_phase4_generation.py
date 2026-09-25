from __future__ import annotations

from pathlib import Path

from agents.phase4_graph import Phase4Orchestrator, build_phase4_graph
from app.schemas import Chunk, IngestionResult, SourceElement
from app.sessions import InMemorySessionStore, UserSessionManager
from generation.crr import CanonicalResponse, TransformationConfig
from generation.service import GenerationService
from providers.llm import HostedLLMProvider


class FakeRetriever:
    def __init__(self):
        self.docs = [
            {
                "id": "c1",
                "text": "Threat activity increased by 32 percent.",
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


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate_json(self, *, system_prompt, user_prompt, schema_name, json_schema, temperature, max_tokens):
        self.calls.append(schema_name)
        if schema_name == "section_digest":
            if "Recommendations" in user_prompt:
                return {
                    "source_id": "made-up",
                    "section": "made-up",
                    "summary": "MFA is recommended.",
                    "key_points": ["Enable MFA"],
                    "claims": [{"text": "MFA is recommended.", "evidence": ["c2", "ghost"]}],
                }
            return {
                "source_id": "made-up",
                "section": "made-up",
                "summary": "Threat activity rose by 32 percent.",
                "key_points": ["32 percent increase"],
                "claims": [{"text": "Threat activity rose by 32 percent.", "evidence": ["c1"]}],
            }
        return {
            "crr_version": "1.0",
            "artifact_type": "answer" if "grounded question answering" in user_prompt else "executive_summary",
            "title": "Threat Brief",
            "summary": "Threat activity increased by 32 percent and MFA is recommended.",
            "sections": [{"heading": "Summary", "content": "Grounded output", "bullets": []}],
            "key_points": ["32 percent increase"],
            "claims": [
                {"claim_id": "claim_1", "text": "Threat activity increased by 32 percent.", "evidence": ["c1", "nonexistent"]}
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


def make_agent():
    pipeline = FakePipeline()
    llm = FakeLLM()
    generator = GenerationService(llm, group_context_max_chars=10000)
    graph = build_phase4_graph(pipeline, generator, default_top_k=5, context_max_chars=10000)
    sessions = UserSessionManager(InMemorySessionStore())
    agent = Phase4Orchestrator(graph, session_manager=sessions, pipeline=pipeline)
    return agent, llm


def test_phase4_qa_produces_schema_valid_crr(tmp_path: Path):
    source = tmp_path / "report.txt"
    source.write_text("source", encoding="utf-8")
    agent, llm = make_agent()
    state = agent.invoke(
        user_id="u1",
        session_id="s1",
        source_paths=[str(source)],
        query="What increased?",
        request_mode="qa",
    )
    assert state["status"] == "phase4_complete"
    crr = CanonicalResponse.model_validate(state["canonical_response"])
    assert crr.artifact_type == "answer"
    assert crr.claims[0].evidence == ["c1"]
    assert any("unknown evidence" in warning for warning in state["warnings"])
    assert state["generation_metadata"]["strategy"] == "single_pass_grounded_qa"
    assert llm.calls == ["canonical_response"]


def test_phase4_transform_is_hierarchical(tmp_path: Path):
    source = tmp_path / "report.txt"
    source.write_text("source", encoding="utf-8")
    agent, llm = make_agent()
    state = agent.invoke(
        user_id="u1",
        session_id="s1",
        source_paths=[str(source)],
        query="Turn the complete report into an executive summary",
        request_mode="transform",
        transformation_config={"artifact_type": "executive_summary", "audience": "leadership"},
    )
    assert state["status"] == "phase4_complete"
    assert len(state["section_digests"]) == 2
    assert state["section_digests"][0]["source_id"] == "src-1"
    assert {d["section"] for d in state["section_digests"]} == {"Overview", "Recommendations"}
    assert state["generation_metadata"]["llm_calls"] == 3
    assert llm.calls.count("section_digest") == 2
    assert llm.calls[-1] == "canonical_response"
    # invalid digest evidence is removed before synthesis state is persisted
    recommendation = next(d for d in state["section_digests"] if d["section"] == "Recommendations")
    assert recommendation["claims"][0]["evidence"] == ["c2"]


def test_transformation_config_is_open_ended():
    config = TransformationConfig.model_validate({
        "artifact_type": "custom_internal_brief",
        "tone": "measured",
        "audience": "field-operators",
        "language": "Hindi",
        "output_formats": ["text", "pdf", "text"],
        "organization_profile": "example",
    })
    assert config.artifact_type == "custom_internal_brief"
    assert config.output_formats == ["text", "pdf"]
    assert config.model_dump()["organization_profile"] == "example"


def test_llm_json_parser_handles_code_fence():
    data = HostedLLMProvider._parse_json('```json\n{"ok": true}\n```')
    assert data == {"ok": True}
