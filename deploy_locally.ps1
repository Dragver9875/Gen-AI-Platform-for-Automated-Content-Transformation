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

function Write-Ok([string]$Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Fail([string]$Message) {
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    exit 1
}

function Test-PythonVersion {
    param(
        [Parameter(Mandatory=$true)][string]$Command,
        [string[]]$PrefixArgs = @()
    )

    try {
        # Keep a failed native probe from terminating this script under
        # $ErrorActionPreference = 'Stop'.
        $oldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $versionText = & $Command @PrefixArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $oldPreference
        }

        if ($exitCode -ne 0 -or -not $versionText) {
            return $null
        }

        $version = [Version]($versionText.Trim())
        if ($version -lt [Version]"3.11.0") {
            return $null
        }

        return $version
    }
    catch {
        return $null
    }
}

function Get-BootstrapPython {
    # 1) Prefer the interpreter backing the currently active virtual environment.
    #    This is the most reliable choice when the user launched this script from
    #    an already-working project shell, e.g. '(.venv) PS ...>'.
    if ($env:VIRTUAL_ENV) {
        $activePython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (Test-Path $activePython) {
            $version = Test-PythonVersion -Command $activePython
            if ($version) {
                return [PSCustomObject]@{
                    Command = $activePython
                    PrefixArgs = @()
                    Version = $version
                    Source = "active virtual environment"
                }
            }
        }
    }

    # 2) Prefer python/python3 from PATH before touching the Windows py launcher.
    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $exe = if ($cmd.Source) { $cmd.Source } else { $name }
            $version = Test-PythonVersion -Command $exe
            if ($version) {
                return [PSCustomObject]@{
                    Command = $exe
                    PrefixArgs = @()
                    Version = $version
                    Source = "$name on PATH"
                }
            }
        }
    }

    # 3) Only then try the Windows launcher. Missing registered runtimes are
    #    treated as a normal failed probe rather than a fatal PowerShell error.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($flag in @("-3.13", "-3.12", "-3.11", "-3")) {
            $version = Test-PythonVersion -Command "py" -PrefixArgs @($flag)
            if ($version) {
                return [PSCustomObject]@{
                    Command = "py"
                    PrefixArgs = @($flag)
                    Version = $version
                    Source = "Windows py launcher $flag"
                }
            }
        }
    }

    return $null
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
    Write-Step "Selecting Python 3.11+ bootstrap interpreter"
    $bootstrap = Get-BootstrapPython
    if (-not $bootstrap) {
        Fail @"
Python 3.11+ could not be located.

If Python is already installed, verify one of these works:
  python --version
  py -0p

Otherwise install Python 3.12, reopen PowerShell, and rerun:
  winget install --id Python.Python.3.12 -e --source winget
"@
    }

    Write-Ok "Using $($bootstrap.Command) $($bootstrap.PrefixArgs -join ' ') (Python $($bootstrap.Version), $($bootstrap.Source))"

    Write-Step "Creating isolated local GUI virtual environment"
    try {
        & $bootstrap.Command @($bootstrap.PrefixArgs) -m venv $VenvDir
    }
    catch {
        Fail "Failed to create .venv_local using Python $($bootstrap.Version): $($_.Exception.Message)"
    }

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        Fail "Python was found, but creating .venv_local failed. Try: python -m venv .venv_local"
    }
}
else {
    $localVersion = Test-PythonVersion -Command $VenvPython
    if (-not $localVersion) {
        Fail ".venv_local exists but its Python runtime is invalid or older than 3.11. Delete .venv_local and rerun."
    }
    Write-Ok "Reusing .venv_local (Python $localVersion)"
}

if (-not $SkipInstall) {
    Write-Step "Installing Python dependencies"
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed." }

    if (-not (Test-Path "requirements-local-gui.txt")) {
        Fail "requirements-local-gui.txt is missing. Use the cleaned repository supplied with deploy_locally.ps1."
    }
    & $VenvPython -m pip install -r requirements-local-gui.txt
    if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed." }

    if ($UsePostgres) {
        if (-not (Test-Path "requirements-postgres.txt")) {
            Fail "requirements-postgres.txt is missing."
        }
        & $VenvPython -m pip install -r requirements-postgres.txt
        if ($LASTEXITCODE -ne 0) { Fail "PostgreSQL dependency installation failed." }
    }
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
print('Creative image output:', 'enabled via HF_TOKEN / FLUX' if os.environ.get('HF_TOKEN', '').strip() else 'disabled')
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
    & $VenvPython -m pytest -q tests/test_phase2_ingestion.py tests/test_hf_token_config.py tests/test_hf_vlm_reranker.py tests/test_phase6_artifacts.py
    if ($LASTEXITCODE -ne 0) { Fail "Smoke tests failed. Fix the test failures before benchmarking live providers." }
}

Write-Step "Launching multimodal GUI on http://127.0.0.1:$Port"
Write-Host "The GUI calls build_phase6() directly and is the supported local multimodal entrypoint." -ForegroundColor DarkGray

# Refuse to start on an occupied port rather than launching a second process that
# immediately exits and leaves the browser showing ERR_CONNECTION_REFUSED.
$existingListener = $null
try {
    $existingListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
}
catch {
    # Get-NetTCPConnection is unavailable on some older Windows builds; the
    # Streamlit health probe below remains authoritative.
}
if ($existingListener) {
    Fail "Port $Port is already in use by PID $($existingListener.OwningProcess). Choose another port, e.g. .\deploy_locally.ps1 -Port 8502"
}

$RuntimeLogDir = Join-Path $RepoRoot ".runtime_logs"
New-Item -ItemType Directory -Path $RuntimeLogDir -Force | Out-Null
$StdoutLog = Join-Path $RuntimeLogDir "streamlit.stdout.log"
$StderrLog = Join-Path $RuntimeLogDir "streamlit.stderr.log"
Remove-Item $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue

$StreamlitArgs = @(
    "-m", "streamlit", "run", "app/local_gui.py",
    "--server.address", "127.0.0.1",
    "--server.port", "$Port",
    "--server.headless", "true",
    "--server.maxUploadSize", "100",
    "--browser.gatherUsageStats", "false"
)

try {
    $GuiProcess = Start-Process `
        -FilePath $VenvPython `
        -ArgumentList $StreamlitArgs `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -PassThru `
        -WindowStyle Hidden
}
catch {
    Fail "Failed to start Streamlit: $($_.Exception.Message)"
}

$HealthUrl = "http://127.0.0.1:$Port/_stcore/health"
$GuiUrl = "http://127.0.0.1:$Port"
$Ready = $false
$StartupDeadline = (Get-Date).AddSeconds(45)

Write-Host "Waiting for Streamlit health endpoint..." -ForegroundColor DarkGray
while ((Get-Date) -lt $StartupDeadline) {
    if ($GuiProcess.HasExited) {
        break
    }

    try {
        $health = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($health.StatusCode -eq 200) {
            $Ready = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 750
    }
}

if (-not $Ready) {
    Write-Host "`n[ERROR] Streamlit did not become reachable on $GuiUrl" -ForegroundColor Red
    if ($GuiProcess.HasExited) {
        Write-Host "Process exited with code $($GuiProcess.ExitCode)." -ForegroundColor Red
    }
    else {
        Write-Host "Process is still running but the health endpoint is not responding." -ForegroundColor Yellow
    }

    if (Test-Path $StderrLog) {
        $stderrText = Get-Content $StderrLog -Raw -ErrorAction SilentlyContinue
        if ($stderrText) {
            Write-Host "`n--- Streamlit stderr ---" -ForegroundColor Yellow
            Write-Host $stderrText
        }
    }
    if (Test-Path $StdoutLog) {
        $stdoutText = Get-Content $StdoutLog -Raw -ErrorAction SilentlyContinue
        if ($stdoutText) {
            Write-Host "`n--- Streamlit stdout ---" -ForegroundColor Yellow
            Write-Host $stdoutText
        }
    }

    Write-Host "`nLogs:" -ForegroundColor Yellow
    Write-Host "  $StdoutLog"
    Write-Host "  $StderrLog"

    if (-not $GuiProcess.HasExited) {
        try { Stop-Process -Id $GuiProcess.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
    exit 1
}

Write-Ok "Multimodal GUI is healthy"
Write-Host "URL:  $GuiUrl" -ForegroundColor Green
Write-Host "PID:  $($GuiProcess.Id)" -ForegroundColor DarkGray
Write-Host "Logs: $RuntimeLogDir" -ForegroundColor DarkGray
Write-Host "Press Ctrl+C here to stop the GUI." -ForegroundColor DarkGray

if (-not $NoBrowser) {
    try {
        Start-Process $GuiUrl | Out-Null
    }
    catch {
        Write-Warn "Could not open the browser automatically. Open $GuiUrl manually."
    }
}

try {
    Wait-Process -Id $GuiProcess.Id
}
finally {
    if (-not $GuiProcess.HasExited) {
        Stop-Process -Id $GuiProcess.Id -Force -ErrorAction SilentlyContinue
    }
}

exit $GuiProcess.ExitCode
