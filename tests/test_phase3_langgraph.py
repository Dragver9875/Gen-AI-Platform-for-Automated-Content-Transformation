from __future__ import annotations

from pathlib import Path

from agents.graph import Phase3Orchestrator, build_phase3_graph
from app.schemas import Chunk, IngestionResult, SourceElement


class FakeRetriever:
    def __init__(self):
        self.retrieve_calls = []
        self.corpus_calls = []
        self.docs = [
            {
                "id": "c1",
                "text": "Threat activity increased by 32 percent.",
                "metadata": {
                    "chunk_id": "c1",
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
        self.retrieve_calls.append({"query": query, "where": where, "final_k": final_k})
        return self.docs[:final_k]

    def get_corpus(self, *, where=None, limit=None):
        self.corpus_calls.append({"where": where, "limit": limit})
        return self.docs[:limit] if limit else list(self.docs)


class FakePipeline:
    def __init__(self):
        self.retriever = FakeRetriever()
        self.ingest_calls = []

    def ingest_and_index(self, path, *, session_id):
        self.ingest_calls.append((str(path), session_id))
        result = IngestionResult(
            source_id="src-1",
            filename=Path(path).name,
            media_type="text/plain",
            strategy="text-direct",
            elements=[SourceElement("e1", "paragraph", "hello")],
        )
        chunk = Chunk(
            "c1",
            "hello",
            {
                "chunk_id": "c1",
                "session_id": session_id,
                "source_id": "src-1",
                "filename": Path(path).name,
            },
        )
        return result, [chunk]


def make_agent() -> tuple[Phase3Orchestrator, FakePipeline]:
    pipeline = FakePipeline()
    graph = build_phase3_graph(pipeline, default_top_k=5, context_max_chars=10000)
    return Phase3Orchestrator(graph), pipeline


def test_index_only_then_resume_same_session_for_qa(tmp_path: Path):
    file = tmp_path / "report.txt"
    file.write_text("source", encoding="utf-8")
    agent, pipeline = make_agent()

    indexed = agent.invoke(session_id="s1", source_paths=[str(file)])
    assert indexed["status"] == "indexed_ready"
    assert indexed["detected_intent"] == "index_only"
    assert indexed["active_source_ids"] == ["src-1"]
    assert indexed["pending_source_paths"] == []

    answered = agent.invoke(session_id="s1", query="What increased?")
    assert answered["status"] == "phase4_ready"
    assert answered["detected_intent"] == "qa"
    assert answered["retrieval_mode"] == "hybrid_top_k"
    assert len(answered["retrieved_documents"]) == 2
    assert "Threat activity increased" in answered["prepared_context"]
    assert len(pipeline.ingest_calls) == 1, "checkpoint resume must not re-ingest the old file"


def test_transform_routes_to_full_document_hierarchy(tmp_path: Path):
    file = tmp_path / "report.txt"
    file.write_text("source", encoding="utf-8")
    agent, pipeline = make_agent()

    state = agent.invoke(
        session_id="s1",
        source_paths=[str(file)],
        query="Summarize this complete report into an executive briefing",
    )
    assert state["detected_intent"] == "transform"
    assert state["retrieval_mode"] == "hierarchical_full_document"
    assert len(state["context_groups"]) == 2
    assert {g["section"] for g in state["context_groups"]} == {"Overview", "Recommendations"}
    assert pipeline.retriever.corpus_calls
    assert not pipeline.retriever.retrieve_calls


def test_explicit_mode_overrides_heuristic(tmp_path: Path):
    file = tmp_path / "report.txt"
    file.write_text("source", encoding="utf-8")
    agent, _ = make_agent()
    state = agent.invoke(
        session_id="s1",
        source_paths=[str(file)],
        query="Summarize the recommendations?",
        request_mode="qa",
    )
    assert state["detected_intent"] == "qa"
    assert state["retrieval_mode"] == "hybrid_top_k"


def test_missing_source_returns_error():
    agent, _ = make_agent()
    state = agent.invoke(session_id="empty", query="What does the report say?")
    assert state["status"] == "error"
    assert state["detected_intent"] == "error"
    assert any("No indexed sources" in err for err in state["errors"])


def test_missing_file_returns_error(tmp_path: Path):
    agent, _ = make_agent()
    state = agent.invoke(session_id="s1", source_paths=[str(tmp_path / "missing.pdf")])
    assert state["status"] == "error"
    assert any("Source file not found" in err for err in state["errors"])


def test_artifact_word_inside_question_stays_qa(tmp_path: Path):
    file = tmp_path / "report.txt"
    file.write_text("source", encoding="utf-8")
    agent, _ = make_agent()
    state = agent.invoke(
        session_id="s1",
        source_paths=[str(file)],
        query="What does the executive summary say?",
    )
    assert state["detected_intent"] == "qa"
