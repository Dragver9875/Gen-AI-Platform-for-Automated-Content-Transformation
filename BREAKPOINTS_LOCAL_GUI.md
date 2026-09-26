# Updated repository: breakpoints found during local multimodal review

## Critical

1. **The React attachment UI does not upload file bytes.** `frontend/src/App.jsx` stores only the selected file name and size, then sends JSON containing `file_name` to `/api/transform`. The backend therefore cannot ingest the attached PDF/PPTX/image.
2. **The React `/api/transform` path bypasses Phase 6.** `app/server.py` defines `get_agent()` / `build_phase6()`, but `/api/transform` calls a separate `call_huggingface_llm(req.query, ...)` function instead. Docling, SigLIP/VLM routing, Harrier, Chroma retrieval, verification and the artifact registry are skipped.
3. **Synthetic fallback output can hide provider failures.** If the direct LLM fails or returns invalid JSON, `generate_contextual_artifacts()` creates plausible-looking output, including security values/metrics. A GUI response is therefore not proof the AI pipeline worked.
4. **The uploaded archive contains a populated `.env`.** It is git-ignored, but packaging the working directory can leak live credentials. Do not distribute archives containing `.env`; rotate secrets if this archive was shared outside the intended team.

## High

5. **Clean API deployment dependencies are incomplete.** `app/server.py` imports FastAPI, Uvicorn and Requests, and multipart support would be needed for uploads, but these packages are absent from `requirements.txt`.
6. **Frontend/backend model configuration diverges from the real pipeline.** `app/server.py` has its own direct-LLM default/fallback logic instead of consuming `Settings` + the provider registry. This can reintroduce model/router failures already solved in the Phase 6 path.
7. **PDF routing is more expensive than the strategy label implies.** `providers/docling.py` renders every PDF page and sends every page image to Granite Docling even when preflight says `native-text`; the `do_ocr`/`force_ocr` flags are intentionally ignored. On mixed/scanned pages the router may then run an additional VLM visual fallback, producing duplicate visual inference.
8. **PPTX ingestion is not fully multimodal.** The HF-only PPTX path extracts text and tables with `python-pptx`, but does not inspect embedded images, charts, SmartArt, or diagrams with the VLM. Visual-heavy slide decks can lose important information.
9. **Synchronous pipeline execution has no job queue.** Large PDFs can require many sequential external calls. Running that directly inside a synchronous web request risks long waits/timeouts. The local Streamlit verifier intentionally blocks so latency is visible, but a production service should use jobs/workers.

## Medium

10. **The current React UI exposes outputs outside the current artifact registry**, including Video Script and X Thread, while the real Phase 6 registry is text/PDF/PPTX/SVG/image. The two product contracts have drifted.
11. **The UI displays misleading static state.** It can show “Uploaded & Indexed” from a filename and “PostgreSQL Connected” from static markup even when those operations did not occur.
12. **Tenant identity is not authenticated.** The browser uses `default_user`, and API requests can supply arbitrary `user_id`. This is unacceptable for production multi-tenant isolation even though the internal vector filters are user/session scoped.
13. **CORS is unrestricted while credentials are enabled.** This should be locked to known origins before deployment.
14. **The repository archive is very large because it contains `.venv` (~hundreds of MB), and a separate `random/` virtual environment has tracked files.** This hurts reproducibility and distribution and should be removed from version control/artifacts.
15. **Live-provider coverage is missing from the automated tests.** The ingestion/provider tests use mocks, so they validate contracts and routing logic, not current Hugging Face model availability or account quota.

## Local verification path added

`deploy_locally.ps1` launches `app/local_gui.py`, which calls the real `build_phase6()` pipeline directly and displays ingestion strategies, end-to-end latency, provider telemetry, verification results, warnings/errors, and downloadable artifacts. This intentionally avoids `app/server.py` until the React API is rewired to the real pipeline.
