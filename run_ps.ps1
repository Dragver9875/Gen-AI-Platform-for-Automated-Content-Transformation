[CmdletBinding()]
param(
    [string]$UserId = "local-user",
    [string]$SessionId,
    [Alias("File")]
    [string[]]$Files = @(),
    [string]$Query,
    [string]$SourceText,
    [ValidateSet("auto", "qa", "transform")]
    [string]$Mode = "auto",
    [string]$ArtifactType = "auto",
    [Alias("Formats")]
    [string[]]$Format = @("text"),
    [string]$Audience = "general",
    [string]$Tone = "professional",
    [string]$Language = "English",
    [ValidateSet("concise", "medium", "detailed")]
    [string]$Detail = "medium",
    [string]$VerificationProfile = "strict",
    [switch]$RunTests,
    [switch]$RunEvals,
    [string]$EvalDataset = "scripts/eval_data/sample.json",
    [string]$EvalOutput = "eval_reports/latest.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Get-DotEnvValue {
    param([string]$Name)

    $processValue = [Environment]::GetEnvironmentVariable($Name)
    if ($processValue) { return $processValue }

    $envPath = Join-Path $PSScriptRoot ".env"
    if (-not (Test-Path $envPath)) { return $null }

    foreach ($line in Get-Content -LiteralPath $envPath) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        if ($trimmed -match "^$([Regex]::Escape($Name))=(.*)$") {
            $value = $Matches[1].Trim()
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            return $value
        }
    }
    return $null
}

function Test-Placeholder {
    param([string]$Value)
    if (-not $Value) { return $true }
    $v = $Value.ToLowerInvariant()
    return (
        $v.Contains("your_") -or
        $v.Contains("your-") -or
        $v.Contains("example") -or
        $v.Contains("placeholder") -or
        $v -eq "hf_your_token_here"
    )
}

$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) {
    $RepoRoot = (Get-Location).Path
}
Set-Location -LiteralPath $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

if (-not (Test-Path (Join-Path $RepoRoot ".env"))) {
    throw ".env is missing. Run .\setup.ps1 first, then populate the required credentials."
}

$required = @("HF_TOKEN", "CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE")
$missing = @()
foreach ($name in $required) {
    $value = Get-DotEnvValue -Name $name
    if (Test-Placeholder -Value $value) {
        $missing += $name
    }
}

$sessionBackend = Get-DotEnvValue -Name "SESSION_STORE_BACKEND"
if (-not $sessionBackend) { $sessionBackend = "memory" }
if ($sessionBackend.ToLowerInvariant() -eq "postgres") {
    $databaseUrl = Get-DotEnvValue -Name "SESSION_DATABASE_URL"
    if (Test-Placeholder -Value $databaseUrl) {
        $missing += "SESSION_DATABASE_URL"
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Missing or placeholder configuration:" -ForegroundColor Red
    foreach ($name in ($missing | Sort-Object -Unique)) {
        Write-Host "  - $name" -ForegroundColor Red
    }
    throw "Update .env before running the pipeline."
}

$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"

if ($RunTests) {
    Write-Step "Running tests"
    & $Python -m pytest -q
    exit $LASTEXITCODE
}

if ($RunEvals) {
    $datasetPath = Resolve-Path -LiteralPath $EvalDataset -ErrorAction Stop
    $outputParent = Split-Path -Parent $EvalOutput
    if ($outputParent -and -not (Test-Path $outputParent)) {
        New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
    }

    Write-Step "Running Phase 9 evaluations"
    & $Python -m scripts.run_evals --dataset $datasetPath.Path --output $EvalOutput
    exit $LASTEXITCODE
}

if (-not $Query) {
    $Query = Read-Host "Enter the query/transformation instruction"
}
if (-not $Query) {
    throw "A query is required unless -RunTests or -RunEvals is used."
}

$resolvedFiles = @()
foreach ($filePath in $Files) {
    if (-not (Test-Path -LiteralPath $filePath)) {
        throw "Input file not found: $filePath"
    }
    $resolvedFiles += (Resolve-Path -LiteralPath $filePath).Path
}

# Free-form prompts are valid source content. If -SourceText is supplied, or
# this is a brand-new prompt-only run with no file/session, materialize the
# text as a temporary .txt source so it flows through normal ingestion/RAG.
$tempSourcePath = $null
$inlineSource = $SourceText
if (-not $inlineSource -and $resolvedFiles.Count -eq 0 -and -not $SessionId) {
    $inlineSource = $Query
    Write-Warn "No file or existing session supplied; treating the prompt as inline source content."
}
if ($inlineSource) {
    $runtimeDir = Join-Path $RepoRoot ".runtime_inputs"
    if (-not (Test-Path $runtimeDir)) {
        New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    }
    $tempSourcePath = Join-Path $runtimeDir ("inline-" + [Guid]::NewGuid().ToString("N") + ".txt")
    [System.IO.File]::WriteAllText($tempSourcePath, $inlineSource, [System.Text.UTF8Encoding]::new($false))
    $resolvedFiles += $tempSourcePath
}

if ($Format.Count -eq 0) {
    $Format = @("text")
}

# Windows PowerShell 5.1 can corrupt native-process arguments that contain
# embedded quotes. Encode the query as UTF-8 Base64 and decode it in Python.
$queryBytes = [System.Text.Encoding]::UTF8.GetBytes($Query)
$queryBase64 = [Convert]::ToBase64String($queryBytes)

$argsList = @(
    "-m", "scripts.run_phase6",
    "--user-id", $UserId,
    "--query-b64", $queryBase64,
    "--mode", $Mode,
    "--artifact-type", $ArtifactType,
    "--audience", $Audience,
    "--tone", $Tone,
    "--language", $Language,
    "--detail", $Detail,
    "--verification-profile", $VerificationProfile
)

if ($SessionId) {
    $argsList += @("--session-id", $SessionId)
}

foreach ($filePath in $resolvedFiles) {
    $argsList += @("--file", $filePath)
}

foreach ($fmt in $Format) {
    if (-not $fmt) { continue }
    # Accept either -Format text,pdf or -Format @("text", "pdf").
    foreach ($item in ($fmt -split ",")) {
        $normalized = $item.Trim()
        if ($normalized) {
            $argsList += @("--format", $normalized)
        }
    }
}

Write-Step "Running Phase 6 pipeline"
Write-Host "User:       $UserId"
if ($SessionId) { Write-Host "Session:    $SessionId" }
Write-Host "Mode:       $Mode"
Write-Host "Formats:    $($Format -join ', ')"
Write-Host "Input files:$([Environment]::NewLine)  $($resolvedFiles -join "`n  ")"
Write-Host "Query:      $Query"

try {
    & $Python @argsList
    $exitCode = $LASTEXITCODE
}
finally {
    if ($tempSourcePath -and (Test-Path -LiteralPath $tempSourcePath)) {
        Remove-Item -LiteralPath $tempSourcePath -Force -ErrorAction SilentlyContinue
    }
}

if ($exitCode -ne 0) {
    Write-Warn "Pipeline exited with code $exitCode"
}
exit $exitCode
