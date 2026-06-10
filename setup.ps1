param(
    [switch]$SkipVoskModel
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$ConfigPath = Join-Path $Root "config.json"
$ExampleConfigPath = Join-Path $Root "config.example.json"
$ModelName = "vosk-model-small-en-us-0.15"
$ModelDir = Join-Path $Root "models\$ModelName"
$ModelZip = Join-Path $Root "models\$ModelName.zip"
$ModelUrl = "https://alphacephei.com/vosk/models/$ModelName.zip"

Set-Location $Root

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

Write-Host "Installing Python dependencies, including Vosk..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

if (-not (Test-Path $ConfigPath)) {
    Write-Host "Creating config.json from config.example.json..."
    Copy-Item -Path $ExampleConfigPath -Destination $ConfigPath
}

if ($SkipVoskModel) {
    Write-Host "Skipping Vosk model download because -SkipVoskModel was provided."
} elseif (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "models") | Out-Null
    if (-not (Test-Path $ModelZip)) {
        Write-Host "Downloading default Vosk voice model..."
        Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelZip
    }
    Write-Host "Extracting default Vosk voice model..."
    Expand-Archive -Path $ModelZip -DestinationPath (Join-Path $Root "models") -Force
}

if (-not $SkipVoskModel -and -not (Test-Path $ModelDir)) {
    throw "Vosk model was not found at $ModelDir after setup."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next:"
Write-Host "  1. Edit config.json if needed."
Write-Host "  2. Run .\run_ui.bat or .\run_listener.bat."
