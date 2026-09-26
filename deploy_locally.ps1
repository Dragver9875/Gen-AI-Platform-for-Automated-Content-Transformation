[CmdletBinding()]
param(
    [int]$Port = 8501,
    [switch]$UsePostgres,
    [switch]$SkipInstall,
    [switch]$RunSmokeTests,
    [switch]$InstallTypst,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Warn([string]$Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Fail([string]$Message) {
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    exit 1
}

Write-Host "OmniTransform local multimodal deployment" -ForegroundColor Green
Write-Host "Repository: $RepoRoot"

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Warn "Created .env from .env.example. Fill in HF_TOKEN and Chroma Cloud values, then rerun this script."
        exit 2
    }
    Fail ".env is missing and .env.example was not found."
}

$VenvDir = Join-Path $RepoRoot ".venv_local"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating isolated local GUI virtual environment"
    $created = $false

    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -c "import sys; print(sys.version)" *> $null
        if ($LASTEXITCODE -eq 0) {
            & py -3.11 -m venv $VenvDir
            $created = ($LASTEXITCODE -eq 0)
        }
        if (-not $created) {
            & py -3.12 -c "import sys; print(sys.version)" *> $null
            if ($LASTEXITCODE -eq 0) {
                & py -3.12 -m venv $VenvDir
                $created = ($LASTEXITCODE -eq 0)
            }
        }
    }

    if (-not $created -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c "import sys; assert sys.version_info >= (3,11), sys.version" *> $null
        if ($LASTEXITCODE -eq 0) {
            & python -m venv $VenvDir
            $created = ($LASTEXITCODE -eq 0)
        }
    }

    if (-not $created -or -not (Test-Path $VenvPython)) {
        Fail "Python 3.11+ is required. Install Python 3.11/3.12 and rerun."
    }
}

if (-not $SkipInstall) {
    Write-Step "Installing Python dependencies"
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed." }

    if (-not (Test-Path "requirements-local-gui.txt")) {
        Fail "requirements-local-gui.txt is missing. Use the patched repository supplied with deploy_locally.ps1."
    }
    & $VenvPython -m pip install -r requirements-local-gui.txt
    if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed." }
}
else {
    Write-Warn "Skipping dependency installation."
}

if ($InstallTypst) {
    Write-Step "Checking Typst"
    if (-not (Get-Command typst -ErrorAction SilentlyContinue)) {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            & winget install --id Typst.Typst -e --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "Typst installation failed. PDF runs will retain .typ source instead of a final PDF."
            }
        }
        else {
            Write-Warn "winget is not available. Install Typst manually if final PDF compilation is required."
        }
    }
}
elseif (-not (Get-Command typst -ErrorAction SilentlyContinue)) {
    Write-Warn "Typst is not installed. PDF generation can still be benchmarked, but final output may be source_only (.typ). Use -InstallTypst to install it with winget."
}

# Use memory sessions for local benchmarking unless the caller explicitly wants PostgreSQL.
if ($UsePostgres) {
    $env:SESSION_STORE_BACKEND = "postgres"
}
else {
    $env:SESSION_STORE_BACKEND = "memory"
}

$env:PYTHONPATH = $RepoRoot
$env:PYTHONUNBUFFERED = "1"

Write-Step "Validating configuration"
$Preflight = @'
import os
from dotenv import load_dotenv
load_dotenv('.env', override=False)
required = ['HF_TOKEN', 'CHROMA_API_KEY', 'CHROMA_TENANT', 'CHROMA_DATABASE']
if os.environ.get('SESSION_STORE_BACKEND', '').lower() == 'postgres':
    required.append('SESSION_DATABASE_URL')
missing = [name for name in required if not os.environ.get(name, '').strip() or os.environ.get(name, '').lower().startswith(('your_', 'replace_', 'changeme'))]
if missing:
    print('Missing or placeholder configuration:')
    for name in missing:
        print(f'  - {name}')
    raise SystemExit(3)
print('Core configuration: OK')
print('Session backend:', os.environ.get('SESSION_STORE_BACKEND', 'memory'))
print('Creative image output:', 'enabled' if os.environ.get('IMAGE_GEN_API_URL', '').strip() else 'disabled (IMAGE_GEN_API_URL empty)')
print('Neural reranker:', 'enabled' if os.environ.get('RERANKER_API_URL', '').strip() else 'disabled (RRF fallback)')
'@
& $VenvPython -c $Preflight
if ($LASTEXITCODE -ne 0) {
    Fail "Configuration validation failed. Update .env and rerun."
}

Write-Step "Compiling project Python files"
& $VenvPython -m compileall -q app agents artifacts core database evaluation generation ingestion providers retrieval scripts verification
if ($LASTEXITCODE -ne 0) { Fail "Python compile check failed." }

if ($RunSmokeTests) {
    Write-Step "Running focused multimodal regression tests"
    & $VenvPython -m pytest -q tests/test_phase2_ingestion.py tests/test_hf_docling_reranker.py tests/test_phase6_artifacts.py
    if ($LASTEXITCODE -ne 0) { Fail "Smoke tests failed. Fix the test failures before benchmarking live providers." }
}

Write-Step "Launching multimodal GUI on http://127.0.0.1:$Port"
Write-Host "The GUI calls build_phase6() directly; it does NOT use app/server.py's synthetic fallback path." -ForegroundColor DarkGray
Write-Host "Press Ctrl+C in this terminal to stop the GUI." -ForegroundColor DarkGray

$Headless = if ($NoBrowser) { "true" } else { "false" }
$StreamlitArgs = @(
    "-m", "streamlit", "run", "app/local_gui.py",
    "--server.address", "127.0.0.1",
    "--server.port", "$Port",
    "--server.headless", $Headless,
    "--server.maxUploadSize", "100",
    "--browser.gatherUsageStats", "false"
)

& $VenvPython @StreamlitArgs
exit $LASTEXITCODE
