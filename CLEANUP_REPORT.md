# Repository cleanup report

## Removed because they could execute a divergent/broken runtime

- `app/server.py`: separate FastAPI demo path with its own direct LLM/fallback logic; it bypassed the real Phase 6 multimodal pipeline.
- `frontend/`: React demo UI whose attachment control did not upload file bytes to Phase 6.
- Root Node/Neon scaffolding: `package.json`, `package-lock.json`, `.neon`, `neon.ts`; unused by the Python multimodal runtime.
- Duplicate `gitignore` (without the leading dot).
- `database/utils.py`: unused backward-compatibility helper; the runtime uses `database/chroma.py`.
- `database/.env.example` and `.streamlit/secrets.toml.example`: duplicate configuration surfaces. `.env` is now the single local configuration source.
- `HOTFIX_README.txt` and stale breakpoint notes tied to removed paths.
- Checked-in runtime input directory.

## Dependency cleanup

- `requirements.txt`: core runtime only.
- `requirements-local-gui.txt`: core + Streamlit + pytest for GUI benchmarking/smoke tests.
- `requirements-postgres.txt`: optional PostgreSQL/LangGraph checkpoint dependencies, installed only when PostgreSQL is requested.

## Source/output namespace fix

- `artifacts/` is the Python source package and is no longer ignored by Git.
- Generated files now default to `runtime_artifacts/` via `PHASE6_OUTPUT_DIR`.
- This prevents clean checkouts/ZIP packaging from accidentally dropping the artifact generator package.

## Safer local defaults

- `.env.example` defaults `SESSION_STORE_BACKEND=memory`.
- `deploy_locally.ps1 -UsePostgres` installs PostgreSQL extras and requires `SESSION_DATABASE_URL`.
- The local GUI deletes temporary uploaded source copies after each run.
- The local GUI rejects total staged input above `LOCAL_GUI_MAX_TOTAL_UPLOAD_MB` (default 100 MiB).

## Intentionally retained

The Phase 3/4/5 modules and graph helpers remain because the Phase 6 graph imports/inherits them. They are historical layers, but they are not dead code in the current topology.

## Multimodal runtime after cleanup

- Granite Docling was removed from the default runtime because the Hub checkpoint is not currently deployed by a Hugging Face Inference Provider.
- Native PDFs now use PyMuPDF only; scanned/visual/mixed pages call the VLM selectively.
- PPTX uses python-pptx for text/tables/chart data and sends embedded pictures to the VLM.
- VLM inference supports ordered fallback models when HF reports model/provider unavailability.
- Live Hugging Face quota and provider availability can still change at runtime.

## Windows launcher hardening

`deploy_locally.ps1` now selects Python in this order: active virtual environment, `python`/`python3` on PATH, then the Windows `py` launcher. Missing `py -3.11` or `py -3.12` registrations are treated as ordinary probe failures and can no longer terminate the deployment script under `$ErrorActionPreference = "Stop"`.
