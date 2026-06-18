param(
    [int]$Port = 9090,
    [int]$RestPort = 8000,
    [string]$Backend = "faster_whisper",
    [string]$Model = "base.en",
    [int]$MaxClients = 2,
    [int]$MaxConnectionTime = 43200
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

Set-Location $Root

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment was not found. Run .\build.bat first."
}

Write-Host "Starting WhisperLive server"
Write-Host "WebSocket: ws://localhost:$Port"
Write-Host "REST: http://localhost:$RestPort/v1"
Write-Host "Backend: $Backend"
Write-Host "Model: $Model"
Write-Host ""

& $VenvPython -m pip show whisper-live | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing whisper-live..."
    & $VenvPython -m pip install whisper-live
}

& $VenvPython .\run_whisperlive_server.py `
    --port $Port `
    --rest_port $RestPort `
    --backend $Backend `
    --max_clients $MaxClients `
    --max_connection_time $MaxConnectionTime `
    -fw $Model
