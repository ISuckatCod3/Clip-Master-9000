param(
    [ValidateSet("Portable", "Exe")]
    [string]$Target = "Portable",
    [switch]$SkipVoskModel
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DistRoot = Join-Path $Root "dist"
$PortableDir = Join-Path $DistRoot "Clip-Master-9000"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$ExeName = "Clip Master 9000"

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
Write-Host "Packaging Clip Master 9000 ($Target)"
Write-Host "===================================="
Write-Host ""

$buildArgs = @()
if ($SkipVoskModel) {
    $buildArgs += "-SkipVoskModel"
}
& (Join-Path $Root "build.ps1") @buildArgs

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null

if ($Target -eq "Portable") {
    if (Test-Path $PortableDir) {
        Remove-Item -Path $PortableDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $PortableDir | Out-Null

    $items = @(
        ".venv",
        "assets",
        "models",
        "config.example.json",
        "requirements.txt",
        "README.md",
        "live_video_interpreter.py",
        "control_panel.py",
        "run_listener.bat",
        "run_ui.bat",
        "run_obs_renamer.bat",
        "run_whisperlive_server.bat",
        "run_whisperlive_server.ps1",
        "run_whisperlive_server.py",
        "save_obs_replay.bat",
        "setup.ps1",
        "setup.bat",
        "build.ps1",
        "build.bat"
    )

    foreach ($item in $items) {
        $source = Join-Path $Root $item
        if (Test-Path $source) {
            Copy-Item -Path $source -Destination $PortableDir -Recurse -Force
        }
    }

    if (Test-Path (Join-Path $Root "config.json")) {
        Copy-Item -Path (Join-Path $Root "config.json") -Destination (Join-Path $PortableDir "config.example.local.json") -Force
    }

    New-Item -ItemType Directory -Force -Path (Join-Path $PortableDir "logs") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $PortableDir "clips") | Out-Null

    $ZipPath = Join-Path $DistRoot "Clip-Master-9000-portable.zip"
    if (Test-Path $ZipPath) {
        Remove-Item -Path $ZipPath -Force
    }
    Compress-Archive -Path $PortableDir -DestinationPath $ZipPath -Force

    Write-Host ""
    Write-Host "Portable package created:"
    Write-Host "  $ZipPath"
    exit 0
}

Write-Host "Installing PyInstaller..."
& $VenvPython -m pip install pyinstaller
Write-Host "Installing WhisperLive for EXE packaging..."
& $VenvPython -m pip install whisper-live

$ExistingExe = Join-Path (Join-Path $DistRoot $ExeName) "$ExeName.exe"
if (Test-Path $ExistingExe) {
    $running = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $ExistingExe }
    if ($running) {
        $ids = ($running | Select-Object -ExpandProperty Id) -join ", "
        throw "Close the running Clip Master 9000 app before packaging. Locked process id(s): $ids"
    }
}

$PyInstallerArgs = @(
    "--noconfirm",
    "--onedir",
    "--name", $ExeName,
    "--collect-all", "vosk",
    "--collect-all", "faster_whisper",
    "--collect-all", "ctranslate2",
    "--collect-all", "tokenizers",
    "--collect-all", "onnxruntime",
    "--collect-all", "av",
    "--collect-all", "whisper_live",
    "--add-data", "config.example.json;.",
    "--add-data", "run_whisperlive_server.bat;.",
    "--add-data", "run_whisperlive_server.ps1;.",
    "--add-data", "run_whisperlive_server.py;.",
    "control_panel.py"
)

if (-not $SkipVoskModel -and (Test-Path (Join-Path $Root "models"))) {
    $PyInstallerArgs += @("--add-data", "models;models")
}

if (Test-Path (Join-Path $Root "assets")) {
    $PyInstallerArgs += @("--add-data", "assets;assets")
}

if (Test-Path (Join-Path $Root "assets\app.ico")) {
    $PyInstallerArgs += @("--icon", "assets\app.ico")
}

& $VenvPython -m PyInstaller @PyInstallerArgs

if (Test-Path (Join-Path $Root "config.json")) {
    Copy-Item -Path (Join-Path $Root "config.json") -Destination (Join-Path (Join-Path $DistRoot $ExeName) "config.json") -Force
}

Write-Host ""
Write-Host "Exe package created:"
Write-Host "  $DistRoot\$ExeName\$ExeName.exe"
