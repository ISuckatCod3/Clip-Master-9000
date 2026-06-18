param(
    [switch]$SkipVoskModel,
    [switch]$LaunchUI
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$ConfigPath = Join-Path $Root "config.json"

function Show-ClipMasterLogo {
    Write-Host @'
   ____ _ _        __  __           _              ___   ___   ___   ___
  / ___| (_)_ __  |  \/  | __ _ ___| |_ ___ _ __  / _ \ / _ \ / _ \ / _ \
 | |   | | | '_ \ | |\/| |/ _` / __| __/ _ \ '__| \_, /| | | | | | | | | |
 | |___| | | |_) || |  | | (_| \__ \ ||  __/ |      / / | |_| | |_| | |_| |
  \____|_|_| .__/ |_|  |_|\__,_|___/\__\___|_|     /_/   \___/ \___/ \___/
           |_|
'@
    Write-Host ""
}

Set-Location $Root

Show-ClipMasterLogo
Write-Host "Clip Master 9000 build"
Write-Host "======================"
Write-Host ""

Write-Host "Checking Python..."
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
Write-Host "Python $pythonVersion"

Write-Host ""
Write-Host "Running setup..."
$setupArgs = @()
if ($SkipVoskModel) {
    $setupArgs += "-SkipVoskModel"
}
& (Join-Path $Root "setup.ps1") @setupArgs

Write-Host ""
Write-Host "Creating runtime folders..."
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "clips") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root ".capture_buffer") | Out-Null

Write-Host ""
Write-Host "Validating Python files..."
& $VenvPython -m py_compile live_video_interpreter.py control_panel.py

Write-Host ""
Write-Host "Validating config..."
& $VenvPython -m json.tool config.example.json | Out-Null
if (-not (Test-Path $ConfigPath)) {
    throw "config.json was not created."
}
& $VenvPython -m json.tool config.json | Out-Null

$config = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
if (-not $SkipVoskModel) {
    $modelPath = Join-Path $Root $config.voice.vosk_model_path
    if (-not (Test-Path $modelPath)) {
        throw "Vosk model is missing: $modelPath"
    }
}

Write-Host ""
Write-Host "Checking installed packages..."
& $VenvPython -c "import cv2, faster_whisper, mss, numpy, obsws_python, openai, sounddevice, vosk; print('imports ok')"

Write-Host ""
Write-Host "Build complete."
Write-Host "This command validates the source checkout; it does not refresh files under dist\."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Start OBS and enable WebSocket."
Write-Host "  2. Edit config.json if OBS is not on localhost:4455."
Write-Host "  3. Run .\run_ui.bat to pick audio devices."
Write-Host "  4. Run .\run_listener.bat to start voice commands."
Write-Host "  5. To update packaged output, run .\package.ps1 -Target Portable or .\package.ps1 -Target Exe."

if ($LaunchUI) {
    Write-Host ""
    Write-Host "Launching UI..."
    & (Join-Path $Root "run_ui.bat")
}
