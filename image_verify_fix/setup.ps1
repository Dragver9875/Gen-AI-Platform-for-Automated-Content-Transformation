[CmdletBinding()]
param(
    [switch]$ForceRecreateVenv,
    [switch]$SkipTypst,
    [switch]$SkipTests,
    [switch]$UseMemorySessions,
    [switch]$UsePostgresSessions
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Get-PythonCandidate {
    $candidates = @()

    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += ,@("py", "-3.12")
        $candidates += ,@("py", "-3.11")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates += ,@("python")
    }
    if (Get-Command python3 -ErrorAction SilentlyContinue) {
        $candidates += ,@("python3")
    }

    foreach ($candidate in $candidates) {
        $command = $candidate[0]
        $prefix = @()
        if ($candidate.Count -gt 1) {
            $prefix = @($candidate[1..($candidate.Count - 1)])
        }

        try {
            $versionText = & $command @prefix -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
            if (-not $versionText) { continue }
            $version = [Version]($versionText.Trim())
            if ($version -ge [Version]"3.11.0") {
                return [PSCustomObject]@{
                    Command = $command
                    PrefixArgs = $prefix
                    Version = $version
                }
            }
        }
        catch {
            # Try the next candidate.
        }
    }

    return $null
}

function Set-DotEnvValue {
    param(
        [string]$Path,
        [string]$Name,
        [string]$Value
    )

    $content = if (Test-Path $Path) { Get-Content -LiteralPath $Path -Raw } else { "" }
    $escapedName = [Regex]::Escape($Name)
    $pattern = "(?m)^$escapedName=.*$"
    $replacement = "$Name=$Value"

    if ($content -match $pattern) {
        $content = [Regex]::Replace($content, $pattern, $replacement)
    }
    else {
        if ($content.Length -gt 0 -and -not $content.EndsWith("`n")) {
            $content += "`r`n"
        }
        $content += "$replacement`r`n"
    }

    Set-Content -LiteralPath $Path -Value $content -Encoding UTF8
}

$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) {
    $RepoRoot = (Get-Location).Path
}
Set-Location -LiteralPath $RepoRoot

if ($UseMemorySessions -and $UsePostgresSessions) {
    throw "Choose only one of -UseMemorySessions or -UsePostgresSessions."
}

Write-Host "TheName - Windows setup" -ForegroundColor White
Write-Host "Repository: $RepoRoot"

if (-not (Test-Path "requirements.txt")) {
    throw "requirements.txt was not found in $RepoRoot. Place setup.ps1 in the repository root and run it again."
}

Write-Step "Checking Python"
$python = Get-PythonCandidate
if (-not $python) {
    throw @"
Python 3.11+ was not found.
Install Python first, for example:
  winget install --id Python.Python.3.12 -e --source winget
Then reopen PowerShell and rerun setup.ps1.
"@
}
Write-Ok "Using $($python.Command) $($python.PrefixArgs -join ' ') (Python $($python.Version))"

$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if ($ForceRecreateVenv -and (Test-Path $VenvDir)) {
    Write-Step "Removing existing virtual environment"
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}

if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating .venv"
    & $python.Command @($python.PrefixArgs) -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the virtual environment."
    }
}
else {
    Write-Ok ".venv already exists"
}

Write-Step "Upgrading pip tooling"
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip tooling." }

Write-Step "Installing Python dependencies"
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
Write-Ok "Python dependencies installed"

$EnvPath = Join-Path $RepoRoot ".env"
$EnvExamplePath = Join-Path $RepoRoot ".env.example"

Write-Step "Preparing environment configuration"
if (-not (Test-Path $EnvPath)) {
    if (-not (Test-Path $EnvExamplePath)) {
        throw ".env.example is missing."
    }
    Copy-Item -LiteralPath $EnvExamplePath -Destination $EnvPath
    Write-Ok "Created .env from .env.example"
}
else {
    Write-Ok ".env already exists; it was not overwritten"
}

if ($UseMemorySessions) {
    Set-DotEnvValue -Path $EnvPath -Name "SESSION_STORE_BACKEND" -Value "memory"
    Write-Warn "SESSION_STORE_BACKEND was set to memory. This is suitable for local development only."
}
if ($UsePostgresSessions) {
    Set-DotEnvValue -Path $EnvPath -Name "SESSION_STORE_BACKEND" -Value "postgres"
}

$sessionBackendLine = Select-String -Path $EnvPath -Pattern '^SESSION_STORE_BACKEND=(.+)$' | Select-Object -First 1
$sessionBackend = if ($sessionBackendLine) { $sessionBackendLine.Matches[0].Groups[1].Value.Trim().ToLowerInvariant() } else { "memory" }
if ($sessionBackend -eq "postgres") {
    Write-Step "Installing optional PostgreSQL persistence dependencies"
    & $VenvPython -m pip install -r requirements-postgres.txt
    if ($LASTEXITCODE -ne 0) { throw "PostgreSQL dependency installation failed." }
}

$ArtifactsDir = Join-Path $RepoRoot "runtime_artifacts"
if (-not (Test-Path $ArtifactsDir)) {
    New-Item -ItemType Directory -Path $ArtifactsDir | Out-Null
}

if (-not $SkipTypst) {
    Write-Step "Checking Typst (required for compiled PDF artifacts)"
    $typst = Get-Command typst -ErrorAction SilentlyContinue
    if (-not $typst) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            Write-Host "Typst was not found. Installing Typst.Typst using WinGet..."
            try {
                & winget install --id Typst.Typst -e --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
                if ($LASTEXITCODE -ne 0) {
                    Write-Warn "WinGet could not install Typst. PDF generation will still retain .typ source, but compiled PDF artifacts require Typst."
                }
            }
            catch {
                Write-Warn "Typst installation failed: $($_.Exception.Message)"
            }
        }
        else {
            Write-Warn "WinGet is unavailable. Install Typst manually and ensure 'typst' is on PATH."
        }
    }
    else {
        Write-Ok "Typst found at $($typst.Source)"
    }
}

Write-Step "Running import smoke check"
& $VenvPython -c "from app.config import Settings; import agents, ingestion, retrieval, providers; print('Core imports OK')"
if ($LASTEXITCODE -ne 0) {
    throw "Core import smoke check failed."
}

if (-not $SkipTests) {
    Write-Step "Installing test dependency"
    & $VenvPython -m pip install "pytest>=8,<9"
    if ($LASTEXITCODE -ne 0) { throw "pytest installation failed." }
    Write-Step "Running test suite"
    & $VenvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        throw "Tests failed. Review the output above before running the project."
    }
    Write-Ok "Tests passed"
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Edit .env and set at minimum:"
Write-Host "       HF_TOKEN"
Write-Host "       CHROMA_API_KEY"
Write-Host "       CHROMA_TENANT"
Write-Host "       CHROMA_DATABASE"
if (-not $UseMemorySessions) {
    Write-Host "       SESSION_DATABASE_URL  (when SESSION_STORE_BACKEND=postgres)"
}
Write-Host "  2. Run the pipeline, for example:"
Write-Host '       .\run_ps.ps1 -UserId user-1 -File .\sample.pdf -Query "Create an executive advisory" -Mode transform -Format text,pdf'
Write-Host ""
