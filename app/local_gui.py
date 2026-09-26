from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from app.config import ConfigurationError, Settings
from app.factory import build_phase6
from core.telemetry import telemetry_session


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = ROOT / ".local_gui_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env", override=False)

SUPPORTED_UPLOAD_SUFFIXES = {
    ".txt", ".md", ".pdf", ".pptx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
}


def _safe_name(name: str) -> str:
    base = Path(name).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(base).stem).strip("-._") or "source"
    suffix = Path(base).suffix.lower()
    return f"{stem[:80]}{suffix}"


def _save_uploads(uploaded_files: list[Any], source_text: str, run_id: str) -> list[str]:
    run_dir = UPLOAD_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    used: set[str] = set()

    for item in uploaded_files:
        name = _safe_name(item.name)
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            raise ValueError(f"Unsupported upload type: {suffix or item.name}")
        candidate = name
        index = 2
        while candidate in used or (run_dir / candidate).exists():
            candidate = f"{Path(name).stem}-{index}{suffix}"
            index += 1
        used.add(candidate)
        path = run_dir / candidate
        path.write_bytes(item.getvalue())
        paths.append(str(path))

    if source_text.strip():
        inline = run_dir / "inline-source.txt"
        inline.write_text(source_text.strip(), encoding="utf-8")
        paths.append(str(inline))

    return paths


def _artifact_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _clean_state_for_debug(state: dict[str, Any]) -> dict[str, Any]:
    hidden = {"prepared_context", "retrieved_documents", "context_groups"}
    return {key: value for key, value in state.items() if key not in hidden}


def _settings_summary(settings: Settings) -> list[dict[str, str]]:
    return [
        {"Capability": "Embeddings", "Provider / model": f"HF · {settings.harrier_model}", "Configured": "yes"},
        {"Capability": "Document understanding", "Provider / model": f"HF · {settings.docling_model}", "Configured": "yes"},
        {"Capability": "Visual routing", "Provider / model": f"HF · {settings.siglip_model}", "Configured": "yes"},
        {"Capability": "Visual understanding", "Provider / model": f"HF · {settings.vlm_model}", "Configured": "yes"},
        {"Capability": "Generation / verifier", "Provider / model": f"HF · {settings.llm_model}", "Configured": "yes" if settings.llm_api_url else "no"},
        {"Capability": "Reranker", "Provider / model": settings.reranker_model, "Configured": "yes" if settings.reranker_api_url else "no (RRF only)"},
        {"Capability": "Creative image", "Provider / model": settings.image_gen_model or "—", "Configured": "yes" if settings.image_gen_api_url else "no"},
        {"Capability": "Session store", "Provider / model": settings.session_store_backend, "Configured": "yes"},
    ]


@st.cache_resource(show_spinner=False)
def _build_agent_cached():
    settings = Settings.from_env()
    return settings, build_phase6(settings)


st.set_page_config(page_title="OmniTransform Multimodal Lab", page_icon="🧪", layout="wide")
st.title("OmniTransform Multimodal Lab")
st.caption("Local GUI for exercising the real Phase 6 LangGraph pipeline — ingestion, routing, retrieval, generation, verification, and artifact creation.")

try:
    settings_preview = Settings.from_env()
except ConfigurationError as exc:
    st.error(f"Configuration error: {exc}")
    st.info("Populate .env with HF_TOKEN, CHROMA_API_KEY, CHROMA_TENANT and CHROMA_DATABASE. For local testing, deploy_locally.ps1 defaults sessions to memory.")
    st.stop()
except Exception as exc:
    st.error(f"Unable to load configuration: {exc}")
    st.stop()

with st.sidebar:
    st.header("Run configuration")
    user_id = st.text_input("User ID", value="local-gui-user")
    request_mode = st.selectbox("Mode", ["auto", "transform", "qa"], index=1)
    artifact_type = st.text_input("Artifact type", value="executive_summary")

    format_options = ["text", "pdf", "pptx", "svg"]
    if settings_preview.image_gen_api_url:
        format_options.append("image")
    output_formats = st.multiselect("Output formats", format_options, default=["text"])

    audience = st.text_input("Audience", value="general")
    tone = st.text_input("Tone", value="professional")
    language = st.text_input("Language", value="English")
    detail = st.select_slider("Detail", options=["brief", "medium", "detailed", "exhaustive"], value="medium")
    verification_profile = st.selectbox("Verification profile", ["strict", "standard", "creative"], index=0)
    top_k = st.slider("QA retrieval top-k", min_value=1, max_value=20, value=5)

    st.divider()
    st.subheader("Resolved capabilities")
    st.dataframe(_settings_summary(settings_preview), hide_index=True, use_container_width=True)
    if not settings_preview.reranker_api_url:
        st.caption("Reranker endpoint is not configured; retrieval will use BM25 + Harrier + RRF without neural reranking.")
    if not settings_preview.image_gen_api_url:
        st.caption("Creative image output is disabled because IMAGE_GEN_API_URL is empty.")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("1. Multimodal sources")
    uploads = st.file_uploader(
        "Upload one or more sources",
        type=["txt", "md", "pdf", "pptx", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        help="Legacy .ppt is intentionally excluded because the HF-only pipeline supports .pptx, not binary .ppt.",
    )
    source_text = st.text_area("Optional inline source text", height=110, placeholder="Paste source text here if you want to combine it with uploaded files.")

    if uploads:
        st.caption(f"{len(uploads)} file(s) staged")
        for item in uploads:
            st.write(f"• `{item.name}` — {len(item.getvalue()) / 1024:.1f} KiB")
            if Path(item.name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                st.image(item.getvalue(), caption=item.name, width=360)

with right:
    st.subheader("2. Transformation / question")
    query = st.text_area(
        "Instruction",
        height=145,
        value="Create a concise professional summary of the uploaded source material.",
    )
    custom_instructions = st.text_area("Optional additional instructions", height=90)
    st.caption("The uploaded files are passed as actual source_paths to Phase 6; this GUI does not use app/server.py's fallback generator.")

run = st.button("Run real multimodal pipeline", type="primary", use_container_width=True)

if run:
    if not uploads and not source_text.strip():
        st.error("Provide at least one uploaded file or inline source text.")
        st.stop()
    if not query.strip():
        st.error("Provide a transformation instruction or question.")
        st.stop()
    if not output_formats:
        st.error("Select at least one output format.")
        st.stop()

    run_id = uuid.uuid4().hex
    source_paths = _save_uploads(list(uploads or []), source_text, run_id)

    with st.status("Running Phase 6 multimodal pipeline…", expanded=True) as status_box:
        try:
            status_box.write("Loading providers and LangGraph…")
            settings, agent = _build_agent_cached()
            status_box.write(f"Sources: {len(source_paths)} · Mode: {request_mode} · Formats: {', '.join(output_formats)}")
            started = time.perf_counter()
            with telemetry_session() as collector:
                state = agent.invoke(
                    user_id=user_id.strip() or "local-gui-user",
                    source_paths=source_paths,
                    query=query.strip(),
                    request_mode=request_mode,
                    top_k=top_k,
                    session_title=f"Local GUI {run_id[:8]}",
                    transformation_config={
                        "artifact_type": artifact_type.strip() or "auto",
                        "output_formats": output_formats,
                        "audience": audience.strip() or "general",
                        "tone": tone.strip() or "professional",
                        "language": language.strip() or "English",
                        "detail_level": detail,
                        "custom_instructions": custom_instructions.strip(),
                        "verification_profile": verification_profile,
                    },
                )
            wall_s = time.perf_counter() - started
            perf = collector.summary()
            perf["wall_time_ms"] = round(wall_s * 1000.0, 2)
            st.session_state["last_run_state"] = dict(state)
            st.session_state["last_run_perf"] = perf
            st.session_state["last_run_paths"] = source_paths
            if state.get("status") == "error":
                status_box.update(label="Pipeline completed with an error", state="error")
            else:
                status_box.update(label=f"Pipeline completed: {state.get('status')}", state="complete")
        except Exception as exc:
            status_box.update(label="Pipeline crashed before returning state", state="error")
            st.exception(exc)
            st.stop()

state = st.session_state.get("last_run_state")
perf = st.session_state.get("last_run_perf")

if state and perf:
    st.divider()
    st.subheader("3. Performance and correctness")
    report = state.get("verification_report") or {}
    ingested = state.get("ingested_sources") or []
    artifacts = state.get("artifacts") or []

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Status", str(state.get("status", "unknown")))
    m2.metric("Wall time", f"{float(perf.get('wall_time_ms', 0.0))/1000:.2f}s")
    m3.metric("Sources", len(ingested))
    m4.metric("Provider calls", int(perf.get("provider_request_count", 0)))
    faith = report.get("faithfulness_score")
    m5.metric("Faithfulness", "—" if faith is None else f"{float(faith):.3f}")

    tab_ingest, tab_provider, tab_verify, tab_artifacts, tab_debug = st.tabs(
        ["Ingestion", "Provider telemetry", "Verification", "Artifacts", "Debug"]
    )

    with tab_ingest:
        if ingested:
            st.dataframe(ingested, hide_index=True, use_container_width=True)
            st.caption("Check `strategy` to confirm whether an image used VLM vs Docling and whether a PDF was classified native/scanned/mixed.")
        else:
            st.warning("No ingested source metadata was returned.")
        if state.get("retrieval_mode"):
            st.write("Retrieval mode:", f"`{state.get('retrieval_mode')}`")
        st.write("Detected intent:", f"`{state.get('detected_intent', 'unknown')}`")

    with tab_provider:
        providers = perf.get("providers") or {}
        rows = []
        for provider, item in providers.items():
            rows.append({"provider": provider, **item})
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No provider telemetry was captured.")
        st.json({key: value for key, value in perf.items() if key != "providers"})

    with tab_verify:
        if report:
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Passed", str(report.get("passed", False)))
            v2.metric("Supported", int(report.get("supported_claims", 0)))
            v3.metric("Partial", int(report.get("partially_supported_claims", 0)))
            v4.metric("Unsupported", int(report.get("unsupported_claims", 0)))
            claims = report.get("claims") or []
            if claims:
                st.dataframe(claims, hide_index=True, use_container_width=True)
        else:
            st.info("No verification report returned (for example, an index-only/error run).")

    with tab_artifacts:
        if not artifacts:
            st.warning("No artifacts returned.")
        for record in artifacts:
            fmt = record.get("format", "artifact")
            status = record.get("status", "unknown")
            with st.container(border=True):
                st.markdown(f"**{fmt.upper()}** · `{status}`")
                if record.get("error"):
                    st.error(record["error"])
                path = _artifact_path(record.get("path"))
                if path and path.exists():
                    data = path.read_bytes()
                    st.caption(str(path))
                    if fmt == "text":
                        st.text_area("Preview", data.decode("utf-8", errors="replace"), height=220, key=f"preview-{record.get('artifact_id')}")
                    elif fmt == "svg":
                        st.image(data)
                    st.download_button(
                        f"Download {record.get('filename') or path.name}",
                        data=data,
                        file_name=record.get("filename") or path.name,
                        mime=record.get("media_type") or "application/octet-stream",
                        key=f"download-{record.get('artifact_id')}",
                    )
                source_only = (record.get("metadata") or {}).get("typst_source_path")
                if source_only:
                    typst_path = _artifact_path(source_only)
                    if typst_path and typst_path.exists():
                        st.download_button(
                            "Download Typst source",
                            data=typst_path.read_bytes(),
                            file_name=typst_path.name,
                            mime="text/plain",
                            key=f"typst-{record.get('artifact_id')}",
                        )

    with tab_debug:
        warnings = state.get("warnings") or []
        errors = state.get("errors") or []
        if warnings:
            st.warning("\n".join(str(x) for x in warnings))
        if errors:
            st.error("\n".join(str(x) for x in errors))
        st.json(_clean_state_for_debug(dict(state)), expanded=False)

st.divider()
st.caption("Local benchmarking note: first-call latency can include provider/model cold starts. Run the same modality at least twice before comparing timings.")
