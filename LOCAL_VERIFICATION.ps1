$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Info([string]$Message) { Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Success([string]$Message) { Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Fail([string]$Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red }

$useComposeV2 = $false

function Invoke-Compose([string[]]$ComposeArgs) {
  if ($script:useComposeV2) {
    & docker compose @ComposeArgs
  } else {
    & docker-compose @ComposeArgs
  }
  if ($LASTEXITCODE -ne 0) {
    throw "Compose command failed ($LASTEXITCODE): $($ComposeArgs -join ' ')"
  }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Fail "Docker not found."
  exit 1
}

try {
  & docker compose version *> $null
  $useComposeV2 = $true
} catch {
  if (Get-Command "docker-compose" -ErrorAction SilentlyContinue) {
    $useComposeV2 = $false
  } else {
    Write-Fail "Docker Compose not found."
    exit 1
  }
}

try { & docker info *> $null } catch { Write-Fail "Docker daemon is not running."; exit 1 }

Write-Info "Containers:"
Invoke-Compose @("-f", "docker-compose.local.yml", "ps")

Write-Host ""
Write-Info "Dependency checks:"
& docker exec cognitionos-postgres-local pg_isready -U cognition_dev -d cognitionos_dev *> $null
if ($LASTEXITCODE -ne 0) { Write-Fail "postgres: FAILED"; exit 1 }
Write-Success "postgres: OK"

& docker exec cognitionos-redis-local redis-cli ping *> $null
if ($LASTEXITCODE -ne 0) { Write-Fail "redis: FAILED"; exit 1 }
Write-Success "redis: OK"

& docker exec cognitionos-rabbitmq-local rabbitmq-diagnostics ping *> $null
if ($LASTEXITCODE -ne 0) { Write-Fail "rabbitmq: FAILED"; exit 1 }
Write-Success "rabbitmq: OK"

function Assert-HttpOk([string]$Url) {
  try {
    $resp = Invoke-WebRequest -Uri $Url -TimeoutSec 5
    if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) { return }
    throw "Unexpected HTTP status: $($resp.StatusCode)"
  } catch {
    throw "HTTP check failed for $Url ($($_.Exception.Message))"
  }
}

Write-Host ""
Write-Info "API checks:"
Assert-HttpOk "http://localhost:8100/health"
Write-Success "/health: OK"
Assert-HttpOk "http://localhost:8100/api/v3/health/system"
Write-Success "/api/v3/health/system: OK"
Assert-HttpOk "http://localhost:8100/api/v3/dashboard"
Write-Success "/api/v3/dashboard: OK"
Assert-HttpOk "http://localhost:8100/api/v3/tasks/active"
Write-Success "/api/v3/tasks/active: OK"

Write-Host ""
Write-Success "Localhost verification passed."
