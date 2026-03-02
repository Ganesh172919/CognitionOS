$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[INFO] Starting CognitionOS localhost stack..."
& "$PSScriptRoot\\scripts\\setup-localhost.ps1"

Write-Host ""
Write-Host "Frontend dashboard:"
Write-Host "  cd frontend; npm install; npm run dev"
Write-Host "  Open: http://localhost:3000"
