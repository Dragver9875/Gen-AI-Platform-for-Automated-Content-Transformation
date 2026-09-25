from __future__ import annotations

from pathlib import Path

from agents.phase6_graph import Phase6Orchestrator, build_phase6_graph
from app.schemas import Chunk, IngestionResult, SourceElement
from app.sessions import InMemorySessionStore, UserSessionManager
from artifacts.generators import (
    CreativeImageArtifactGenerator,
    PDFArtifactGenerator,
    PPTXArtifactGenerator,
    SVGArtifactGenerator,
    TextArtifactGenerator,
)
from artifacts.registry import ArtifactRegistry, ArtifactSpec
from artifacts.serializers import SVGSerializer, TypstPDFSerializer
from artifacts.service import ArtifactService
from generation.service import GenerationService
from verification.service import VerificationService


class FakeRetriever:
    def __init__(self):
        self.docs = [{
            "id": "c1",
            "text": "Threat activity increased by 32 percent during Q3.",
            "metadata": {"chunk_id": "c1", "user_id": "u1", "session_id": "s1", "source_id": "src-1", "filename": "report.txt", "section": "Overview"},
        }]

    def retrieve(self, query, *, where=None, final_k=5, config=None):
        return self.docs[:final_k]

    def get_corpus(self, *, where=None, limit=None):
        return list(self.docs)


class FakePipeline:
    def __init__(self):
        self.retriever = FakeRetriever()

    def ingest_and_index(self, path, *, user_id, session_id):
        result = IngestionResult("src-1", Path(path).name, "text/plain", "text-direct", [SourceElement("e1", "paragraph", "source")])
        return result, [Chunk("c1", "source", {"chunk_id": "c1", "user_id": user_id, "session_id": session_id, "source_id": "src-1"})]


class FakeLLM:
    def generate_json(self, *, system_prompt, user_prompt, schema_name, json_schema, temperature, max_tokens):
        if schema_name == "canonical_response":
            return {
                "crr_version": "1.0", "artifact_type": "executive_summary", "title": "Threat Brief",
                "summary": "Threat activity increased by 32 percent.",
                "sections": [{"heading": "Overview", "content": "Threat activity increased by 32 percent.", "bullets": []}],
                "key_points": ["Increase: 32 percent"],
                "claims": [{"claim_id": "claim_1", "text": "Threat activity increased by 32 percent.", "evidence": ["c1"]}],
                "rendering": {"tone": "professional", "audience": "executive", "language": "English", "detail_level": "medium", "objective": "inform", "style": "clear"},
                "insufficiencies": [],
            }
        if schema_name == "semantic_verification_batch":
            return {"claims": [{"claim_id": "claim_1", "status": "supported", "confidence": 1.0, "supported_evidence": ["c1"], "rationale": "Supported", "unsupported_fragments": [], "suggested_correction": ""}]}
        if schema_name == "text_artifact_ir":
            return {"format": "text", "content": "Threat activity increased by 32 percent.", "extension": ".txt", "media_type": "text/plain"}
        if schema_name == "pdf_artifact_ir":
            return {"format": "pdf", "title": "Threat Brief", "typst_source": "= Threat Brief\n\nThreat activity increased by 32 percent."}
        if schema_name == "presentation_artifact_ir":
            return {"format": "pptx", "title": "Threat Brief", "slides": [{"layout": "title_content", "title": "Overview", "bullets": ["Threat activity increased by 32 percent."], "body": "", "subtitle": "", "speaker_notes": "", "visual_prompt": "", "metrics": []}], "theme": {}}
        if schema_name == "svg_artifact_ir":
            return {"format": "svg", "title": "Threat Brief", "svg": '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400"><text x="20" y="80">32 percent</text></svg>'}
        if schema_name == "creative_image_ir":
            return {"format": "image", "prompt": "Professional abstract cyber threat illustration, no text", "negative_prompt": "text, labels", "width": 512, "height": 512}
        raise AssertionError(schema_name)


class FakeImageProvider:
    def generate(self, prompt, *, negative_prompt="", width=1024, height=1024):
        return b"\x89PNG\r\n\x1a\nFAKE", "image/png"


def make_registry(llm, output_dir):
    registry = ArtifactRegistry()
    registry.register_generator("text", TextArtifactGenerator(llm))
    registry.register_generator("pdf", PDFArtifactGenerator(llm, typst_binary="definitely-missing-typst-binary"))
    registry.register_generator("pptx", PPTXArtifactGenerator(llm))
    registry.register_generator("svg", SVGArtifactGenerator(llm))
    registry.register_generator("creative_image", CreativeImageArtifactGenerator(llm, FakeImageProvider()))
    for spec in [
        ArtifactSpec("text", "text", media_type="text/plain", extension=".txt"),
        ArtifactSpec("pdf", "pdf", media_type="application/pdf", extension=".pdf"),
        ArtifactSpec("pptx", "pptx", media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation", extension=".pptx"),
        ArtifactSpec("svg", "svg", aliases=("infographic",), media_type="image/svg+xml", extension=".svg"),
        ArtifactSpec("image", "creative_image", media_type="image/png", extension=".png"),
    ]:
        registry.register_spec(spec)
    return registry


def make_agent(tmp_path):
    pipeline = FakePipeline()
    llm = FakeLLM()
    generator = GenerationService(llm)
    verifier = VerificationService(llm, repair_generator=llm, max_repair_attempts=1)
    artifact_service = ArtifactService(make_registry(llm, tmp_path), output_dir=tmp_path / "artifacts", fail_fast=False)
    graph = build_phase6_graph(pipeline, generator, verifier, artifact_service, default_top_k=5, context_max_chars=10000)
    return Phase6Orchestrator(graph, session_manager=UserSessionManager(InMemorySessionStore()), pipeline=pipeline, max_repair_attempts=1)


def test_phase6_generates_registered_artifacts(tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("source", encoding="utf-8")
    agent = make_agent(tmp_path)
    state = agent.invoke(
        user_id="u1", session_id="s1", source_paths=[str(source)], query="Create an executive summary",
        request_mode="qa",
        transformation_config={"artifact_type": "executive_summary", "output_formats": ["text", "pdf", "pptx", "infographic", "image"]},
    )
    assert state["status"] == "phase6_complete"
    artifacts = {item["format"]: item for item in state["artifacts"]}
    assert Path(artifacts["text"]["path"]).exists()
    assert artifacts["pdf"]["status"] == "source_only"
    assert Path(artifacts["pdf"]["metadata"]["typst_source_path"]).exists()
    assert Path(artifacts["pptx"]["path"]).exists()
    assert Path(artifacts["svg"]["path"]).exists()
    assert Path(artifacts["image"]["path"]).read_bytes().startswith(b"\x89PNG")


def test_phase6_task_spec_entrypoint(tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("source", encoding="utf-8")
    agent = make_agent(tmp_path)
    state = agent.invoke_task({
        "user_id": "u1", "session_id": "s1", "source_paths": [str(source)], "query": "What changed?", "request_mode": "qa",
        "retrieval": {"strategy": "dense", "final_k": 3},
        "transformation": {"output_formats": ["text"]},
    })
    assert state["status"] == "phase6_complete"
    assert state["retrieval_config"]["strategy"] == "dense"


def test_generated_typst_blocks_external_resource_access(tmp_path):
    ir = type("IR", (), {"typst_source": '#read("/etc/passwd")'})()
    serializer = TypstPDFSerializer(typst_binary="missing")
    try:
        serializer.serialize(ir, tmp_path / "x.pdf")
    except ValueError as exc:
        assert "forbidden" in str(exc).lower()
    else:
        raise AssertionError("unsafe Typst source was not rejected")


def test_generated_svg_blocks_active_content(tmp_path):
    from artifacts.models import SVGArtifactIR
    serializer = SVGSerializer()
    ir = SVGArtifactIR(format="svg", svg="<svg><script>alert(1)</script></svg>")
    try:
        serializer.serialize(ir, tmp_path / "x.svg")
    except ValueError as exc:
        assert "forbidden" in str(exc).lower()
    else:
        raise AssertionError("unsafe SVG was not rejected")
