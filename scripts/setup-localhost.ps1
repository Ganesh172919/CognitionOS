[CmdletBinding()]
param(
  [switch]$NoBuild,
  [switch]$NoCache,
  [switch]$SkipEnvCopy
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Write-Info([string]$Message) { Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Success([string]$Message) { Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Warn([string]$Message) { Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-Fail([string]$Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red }

function Assert-Command([string]$Name, [string]$Hint) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    Write-Fail "$Name not found. $Hint"
    exit 1
  }
}

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

Write-Info "Checking prerequisites..."
Assert-Command -Name "docker" -Hint "Install Docker Desktop and ensure it is on PATH."

try {
  & docker compose version *> $null
  $useComposeV2 = $true
  Write-Success "Docker Compose detected (docker compose)."
} catch {
  if (Get-Command "docker-compose" -ErrorAction SilentlyContinue) {
    $useComposeV2 = $false
    Write-Success "Docker Compose detected (docker-compose)."
  } else {
    Write-Fail "Docker Compose not available. Install Docker Desktop (Compose v2) or docker-compose."
    exit 1
  }
}

try {
  & docker info *> $null
  Write-Success "Docker daemon is running."
} catch {
  Write-Fail "Docker daemon is not running. Start Docker Desktop and retry."
  exit 1
}

Write-Info "Stopping any existing localhost stack..."
try {
  Invoke-Compose @("-f", "docker-compose.local.yml", "down")
} catch {
  Write-Warn "Cleanup reported an error (continuing): $($_.Exception.Message)"
}

if (-not $SkipEnvCopy) {
  Write-Info "Ensuring .env exists..."
  if (-not (Test-Path ".env")) {
    if (-not (Test-Path ".env.localhost")) {
      Write-Fail ".env.localhost not found."
      exit 1
    }
    Copy-Item ".env.localhost" ".env"
    Write-Success "Created .env from .env.localhost"
  } else {
    Write-Warn ".env already exists; leaving it unchanged."
  }
}

if (-not $NoBuild) {
  Write-Info "Building Docker images..."
  $buildArgs = @("-f", "docker-compose.local.yml", "build")
  if ($NoCache) { $buildArgs += "--no-cache" }
  Invoke-Compose $buildArgs
  Write-Success "Images built."
} else {
  Write-Warn "Skipping build (-NoBuild)."
}

Write-Info "Starting postgres, redis, rabbitmq..."
Invoke-Compose @("-f", "docker-compose.local.yml", "up", "-d", "postgres", "redis", "rabbitmq")

Write-Info "Waiting for PostgreSQL..."
for ($i = 1; $i -le 30; $i++) {
  & docker exec cognitionos-postgres-local pg_isready -U cognition_dev -d cognitionos_dev *> $null
  if ($LASTEXITCODE -eq 0) { break }
  if ($i -eq 30) { Write-Fail "PostgreSQL failed to become ready."; exit 1 }
  Start-Sleep -Seconds 1
}
Write-Success "PostgreSQL is ready."

Write-Info "Waiting for Redis..."
for ($i = 1; $i -le 30; $i++) {
  $pong = & docker exec cognitionos-redis-local redis-cli ping 2>$null
  if ($LASTEXITCODE -eq 0 -and $pong -match "PONG") { break }
  if ($i -eq 30) { Write-Fail "Redis failed to become ready."; exit 1 }
  Start-Sleep -Seconds 1
}
Write-Success "Redis is ready."

Write-Info "Waiting for RabbitMQ..."
for ($i = 1; $i -le 60; $i++) {
  & docker exec cognitionos-rabbitmq-local rabbitmq-diagnostics ping *> $null
  if ($LASTEXITCODE -eq 0) { break }
  if ($i -eq 60) { Write-Fail "RabbitMQ failed to become ready."; exit 1 }
  Start-Sleep -Seconds 2
}
Write-Success "RabbitMQ is ready."

Write-Info "Starting API (migrations run on container startup)..."
Invoke-Compose @("-f", "docker-compose.local.yml", "up", "-d", "api")

Write-Info "Waiting for API..."
$apiReady = $false
for ($i = 1; $i -le 60; $i++) {
  try {
    $resp = Invoke-RestMethod -Uri "http://localhost:8100/api/v3/health/live" -TimeoutSec 2
    if ($resp -and $resp.status -eq "alive") { $apiReady = $true; break }
  } catch {
    # ignore and retry
  }
  Start-Sleep -Seconds 1
}

if ($apiReady) {
  Write-Success "API is ready."
} else {
  Write-Warn "API health check timed out. Check logs with: docker compose -f docker-compose.local.yml logs -f api"
}

Write-Host ""
Write-Success "Setup complete."
Write-Host "  API:          http://localhost:8100"
Write-Host "  API docs:     http://localhost:8100/docs"
Write-Host "  System health:http://localhost:8100/api/v3/health/system"
Write-Host "  RabbitMQ UI:  http://localhost:15672 (guest/guest)"
Write-Host ""
Write-Host "Start the frontend dashboard:"
Write-Host "  cd frontend; npm install; npm run dev"
Write-Host "  Open: http://localhost:3000"
