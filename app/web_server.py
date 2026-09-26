from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, _env
from app.factory import build_phase6
from core.telemetry import telemetry_session


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
UPLOAD_ROOT = Path(os.getenv("WEB_UPLOAD_DIR", str(ROOT / "runtime_uploads")))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env", override=False)

SUPPORTED_SUFFIXES = {
    ".txt", ".md", ".pdf", ".pptx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
}
OUTPUT_FORMATS = [
    {"id": "text", "label": "Text", "hint": "Plain text / Markdown"},
    {"id": "pdf", "label": "PDF", "hint": "Polished PDF via Typst"},
    {"id": "pptx", "label": "PPTX", "hint": "Editable presentation"},
    {"id": "infographic", "label": "Infographic", "hint": "Factual SVG"},
    {"id": "image", "label": "Image", "hint": "Creative FLUX image"},
]

MAX_FILE_BYTES = int(os.getenv("WEB_MAX_FILE_MB", "40")) * 1024 * 1024
MAX_TOTAL_BYTES = int(os.getenv("WEB_MAX_TOTAL_UPLOAD_MB", "100")) * 1024 * 1024
MAX_WORKERS = max(1, int(os.getenv("WEB_MAX_WORKERS", "3")))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return cleaned[:96] or fallback


def _safe_filename(name: str) -> str:
    base = Path(name or "source").name
    suffix = Path(base).suffix.lower()
    stem = _safe_component(Path(base).stem, "source")
    return f"{stem}{suffix}"


def _user_id(client_id: str | None) -> str:
    raw = (client_id or "anonymous").strip()
    return "web-" + _safe_component(raw, "anonymous")[:80]


def _present(*names: str) -> bool:
    return any(bool((_env(name) or "").strip()) for name in names)


def _credential_status() -> dict[str, bool]:
    # Presence only: never expose credential values to the browser.
    return {
        "HF_TOKEN": _present("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
        "CHROMA_API_KEY": _present("CHROMA_API_KEY", "CHROMA_CLOUD_API_KEY"),
        "CHROMA_TENANT": _present("CHROMA_TENANT", "CHROMA_CLOUD_TENANT"),
        "CHROMA_DATABASE": _present("CHROMA_DATABASE", "CHROMA_CLOUD_DATABASE"),
    }


@dataclass
class JobRecord:
    job_id: str
    user_id: str
    chat_id: str
    query: str
    formats: list[str]
    status: str = "queued"
    stage: str = "Queued"
    created_at: str = field(default_factory=_utcnow)
    started_at: str | None = None
    completed_at: str | None = None
    state: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    upload_dir: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "chat_id": self.chat_id,
            "query": self.query,
            "formats": self.formats,
            "status": self.status,
            "stage": self.stage,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}

    def add(self, job: JobRecord) -> JobRecord:
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active_for_chat(self, user_id: str, chat_id: str) -> JobRecord | None:
        with self._lock:
            for job in self._jobs.values():
                if job.user_id == user_id and job.chat_id == chat_id and job.status in {"queued", "running"}:
                    return job
        return None

    def update(self, job_id: str, **values: Any) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in values.items():
                setattr(job, key, value)
            return job


class Runtime:
    def __init__(self):
        self._lock = threading.RLock()
        self._agent = None
        self._settings: Settings | None = None
        self._init_error: str | None = None

    def get(self):
        with self._lock:
            if self._agent is not None:
                return self._settings, self._agent
            try:
                settings = Settings.from_env()
                agent = build_phase6(settings)
                self._settings = settings
                self._agent = agent
                self._init_error = None
                return settings, agent
            except Exception as exc:
                self._init_error = str(exc)
                raise

    @property
    def init_error(self) -> str | None:
        return self._init_error


runtime = Runtime()
jobs = JobStore()
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="omnitransform")
app = FastAPI(title="OmniTransform", version="1.0")


def _session_has_sources(agent, user_id: str, chat_id: str) -> bool:
    try:
        docs = agent.pipeline.retriever.get_corpus(
            where={"$and": [{"user_id": user_id}, {"session_id": chat_id}]},
            limit=1,
        )
        return bool(docs)
    except Exception:
        return False


def _assistant_text(state: dict[str, Any], artifacts: list[dict[str, Any]]) -> str:
    for artifact in artifacts:
        if artifact.get("format") == "text" and artifact.get("status") == "generated" and artifact.get("path"):
            path = Path(str(artifact["path"]))
            if not path.is_absolute():
                path = ROOT / path
            if path.exists():
                return path.read_text(encoding="utf-8", errors="replace")

    crr = state.get("canonical_response") or {}
    parts: list[str] = []
    if crr.get("title"):
        parts.append(str(crr["title"]))
    if crr.get("summary"):
        parts.append(str(crr["summary"]))
    for section in crr.get("sections") or []:
        if section.get("heading"):
            parts.append(str(section["heading"]))
        if section.get("content"):
            parts.append(str(section["content"]))
        for bullet in section.get("bullets") or []:
            parts.append(f"• {bullet}")
    if parts:
        return "\n\n".join(parts)
    if state.get("errors"):
        return "\n".join(str(x) for x in state["errors"])
    return "The request completed, but no text preview was returned."


def _artifact_public(job_id: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        path = item.pop("path", None)
        metadata = dict(item.get("metadata") or {})
        metadata.pop("typst_source_path", None)
        item["metadata"] = metadata
        if path and item.get("status") == "generated":
            item["download_url"] = f"/api/jobs/{job_id}/artifacts/{item.get('artifact_id')}"
        items.append(item)
    return items


def _run_job(job_id: str, source_paths: list[str], request_mode: str, custom_instructions: str) -> None:
    job = jobs.update(job_id, status="running", stage="Understanding sources", started_at=_utcnow())
    try:
        settings, agent = runtime.get()
        paths = list(source_paths)

        # ChatGPT-style text-only first turn: treat the first prompt itself as a source
        # if the session has no indexed material yet. Subsequent turns retrieve from the
        # chat's existing Chroma corpus and do not re-index the question.
        if not paths and not _session_has_sources(agent, job.user_id, job.chat_id):
            inline_dir = Path(job.upload_dir or UPLOAD_ROOT / job.job_id)
            inline_dir.mkdir(parents=True, exist_ok=True)
            inline = inline_dir / "inline-source.txt"
            inline.write_text(job.query, encoding="utf-8")
            paths.append(str(inline))

        jobs.update(job_id, stage="Retrieving and generating")
        started = time.perf_counter()
        with telemetry_session() as collector:
            state = agent.invoke(
                user_id=job.user_id,
                session_id=job.chat_id,
                source_paths=paths,
                query=job.query,
                request_mode=request_mode,
                session_title=job.query[:72],
                transformation_config={
                    "artifact_type": "auto",
                    "output_formats": job.formats,
                    "tone": "professional",
                    "audience": "general",
                    "language": "English",
                    "detail_level": "medium",
                    "custom_instructions": custom_instructions,
                    "verification_profile": "strict",
                },
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        perf = collector.summary()
        perf["wall_time_ms"] = elapsed_ms

        records = [dict(x) for x in (state.get("artifacts") or [])]
        verification = dict(state.get("verification_report") or state.get("verification") or {})
        status = str(state.get("status") or "complete")
        result = {
            "status": status,
            "assistant_text": _assistant_text(dict(state), records),
            "artifacts": _artifact_public(job_id, records),
            "warnings": list(state.get("warnings") or []),
            "errors": list(state.get("errors") or []),
            "verification": {
                "passed": verification.get("passed"),
                "faithfulness_score": verification.get("faithfulness_score"),
                "supported_claims": verification.get("supported_claims"),
                "unsupported_claims": verification.get("unsupported_claims"),
            },
            "performance": {
                "wall_time_ms": elapsed_ms,
                "provider_request_count": perf.get("provider_request_count", 0),
                "provider_failure_count": perf.get("provider_failure_count", 0),
            },
            "ingested_sources": list(state.get("ingested_sources") or []),
        }
        public_status = "error" if status == "error" else ("complete_with_issues" if "issues" in status or result["warnings"] or result["errors"] else "complete")
        jobs.update(
            job_id,
            status=public_status,
            stage="Complete" if public_status != "error" else "Failed",
            completed_at=_utcnow(),
            state=dict(state),
            result=result,
            error="\n".join(result["errors"]) if public_status == "error" else None,
        )
    except Exception as exc:
        jobs.update(
            job_id,
            status="error",
            stage="Failed",
            completed_at=_utcnow(),
            error=str(exc),
            result={"status": "error", "assistant_text": f"Request failed: {exc}", "artifacts": [], "warnings": [], "errors": [str(exc)]},
        )
    finally:
        if job.upload_dir:
            shutil.rmtree(job.upload_dir, ignore_errors=True)


@app.get("/api/health")
def health():
    credentials = _credential_status()
    return {
        "ok": True,
        "service": "omnitransform",
        "workers": MAX_WORKERS,
        "runtime_error": runtime.init_error,
        "credentials_present": credentials,
        "missing_credentials": [key for key, present in credentials.items() if not present],
    }


@app.get("/api/bootstrap")
def bootstrap():
    configured = True
    config_error = None
    try:
        settings = Settings.from_env()
        model_info = {
            "multimodal": settings.multimodal_model,
            "generator": settings.llm_model,
            "embedding": settings.harrier_model,
            "image": settings.image_gen_model,
        }
    except Exception as exc:
        configured = False
        config_error = str(exc)
        model_info = {}
    credentials = _credential_status()
    return {
        "configured": configured,
        "config_error": config_error,
        "credentials_present": credentials,
        "missing_credentials": [key for key, present in credentials.items() if not present],
        "outputs": OUTPUT_FORMATS,
        "supported_uploads": sorted(SUPPORTED_SUFFIXES),
        "max_file_mb": MAX_FILE_BYTES // (1024 * 1024),
        "max_total_upload_mb": MAX_TOTAL_BYTES // (1024 * 1024),
        "max_parallel_jobs": MAX_WORKERS,
        "models": model_info,
    }


@app.post("/api/chats")
def create_chat(x_client_id: str | None = Header(default=None)):
    # Creating an empty chat is a UI/session action, not an inference action.
    # Do not initialize ML providers here: that would make the entire chat UI
    # unusable when a provider credential is temporarily missing.
    user_id = _user_id(x_client_id)
    return {
        "user_id": user_id,
        "session_id": uuid.uuid4().hex,
        "title": "New chat",
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
    }


@app.post("/api/chats/{chat_id}/jobs")
async def create_job(
    chat_id: str,
    query: str = Form(...),
    formats: str = Form("[\"text\"]"),
    request_mode: str = Form("auto"),
    custom_instructions: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    x_client_id: str | None = Header(default=None),
):
    user_id = _user_id(x_client_id)
    chat_id = _safe_component(chat_id, "chat")
    active_job = jobs.active_for_chat(user_id, chat_id)
    if active_job is not None:
        raise HTTPException(status_code=409, detail="This chat already has a request running. Open a new chat to work in parallel.")
    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Message is required")

    try:
        selected_formats = json.loads(formats)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Invalid output format payload") from exc
    if not isinstance(selected_formats, list):
        raise HTTPException(status_code=422, detail="Output formats must be a list")
    allowed = {x["id"] for x in OUTPUT_FORMATS}
    selected_formats = list(dict.fromkeys(str(x).lower() for x in selected_formats if str(x).lower() in allowed)) or ["text"]

    job_id = uuid.uuid4().hex
    upload_dir = UPLOAD_ROOT / user_id / chat_id / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    total = 0
    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix or upload.filename}")
            data = await upload.read()
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB")
            total += len(data)
            if total > MAX_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail=f"Total upload exceeds {MAX_TOTAL_BYTES // (1024 * 1024)} MB")
            name = _safe_filename(upload.filename or "source")
            destination = upload_dir / name
            counter = 2
            while destination.exists():
                destination = upload_dir / f"{Path(name).stem}-{counter}{suffix}"
                counter += 1
            destination.write_bytes(data)
            paths.append(str(destination))
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    # Provider/session initialization happens in the worker. The agent's invoke()
    # path ensures the session exists once the runtime is available.
    job = JobRecord(
        job_id=job_id,
        user_id=user_id,
        chat_id=chat_id,
        query=query,
        formats=selected_formats,
        upload_dir=str(upload_dir),
    )
    jobs.add(job)
    executor.submit(_run_job, job_id, paths, request_mode, custom_instructions)
    return JSONResponse(job.public(), status_code=202)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, x_client_id: str | None = Header(default=None)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    if job.user_id != _user_id(x_client_id):
        raise HTTPException(status_code=404, detail="Unknown job")
    return job.public()


@app.get("/api/jobs/{job_id}/artifacts/{artifact_id}")
def download_artifact(job_id: str, artifact_id: str, x_client_id: str | None = Header(default=None)):
    job = jobs.get(job_id)
    if not job or job.user_id != _user_id(x_client_id) or not job.state:
        raise HTTPException(status_code=404, detail="Artifact not found")
    for record in job.state.get("artifacts") or []:
        if str(record.get("artifact_id")) != artifact_id or record.get("status") != "generated":
            continue
        raw = record.get("path")
        if not raw:
            break
        path = Path(str(raw))
        if not path.is_absolute():
            path = ROOT / path
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError:
            break
        output_root = (ROOT / os.getenv("PHASE6_OUTPUT_DIR", "runtime_artifacts")).resolve()
        if output_root not in path.parents:
            raise HTTPException(status_code=403, detail="Invalid artifact path")
        return FileResponse(path, media_type=record.get("media_type") or "application/octet-stream", filename=record.get("filename") or path.name)
    raise HTTPException(status_code=404, detail="Artifact not found")


# SPA routes must be mounted last so /api/* wins first.
if WEB_ROOT.exists():
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")


@app.get("/{full_path:path}")
def spa(full_path: str):
    index = WEB_ROOT / "index.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail="Frontend assets are missing")
    return FileResponse(index)
