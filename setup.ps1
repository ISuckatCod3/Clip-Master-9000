param(
    [switch]$SkipVoskModel
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$ConfigPath = Join-Path $Root "config.json"
$ExampleConfigPath = Join-Path $Root "config.example.json"
$PreviousDefaultModelPath = "models/vosk-model-small-en-us-0.15"
$ModelName = "vosk-model-en-us-0.22-lgraph"
$DefaultModelPath = "models/$ModelName"
$ModelDir = Join-Path $Root "models\$ModelName"
$ModelZip = Join-Path $Root "models\$ModelName.zip"
$ModelUrl = "https://alphacephei.com/vosk/models/$ModelName.zip"

Set-Location $Root

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

Write-Host "Installing Python dependencies, including Vosk and faster-whisper..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

if (-not (Test-Path $ConfigPath)) {
    Write-Host "Creating config.json from config.example.json..."
    Copy-Item -Path $ExampleConfigPath -Destination $ConfigPath
}

$config = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
if (-not $config.voice) {
    $config | Add-Member -MemberType NoteProperty -Name "voice" -Value ([pscustomobject]@{}) -Force
}
if (-not $config.local_whisper) {
    $config | Add-Member -MemberType NoteProperty -Name "local_whisper" -Value ([pscustomobject]@{}) -Force
}
if (-not $config.rename_transcription_provider -or [string]::IsNullOrWhiteSpace([string]$config.rename_transcription_provider)) {
    $config | Add-Member -MemberType NoteProperty -Name "rename_transcription_provider" -Value "local_whisper" -Force
}
if ($null -eq $config.rename_transcription_audio_fraction) {
    $config | Add-Member -MemberType NoteProperty -Name "rename_transcription_audio_fraction" -Value 0.5 -Force
}
if (-not $config.local_whisper.model_size) {
    $config.local_whisper | Add-Member -MemberType NoteProperty -Name "model_size" -Value "base.en" -Force
}
if (-not $config.local_whisper.device) {
    $config.local_whisper | Add-Member -MemberType NoteProperty -Name "device" -Value "auto" -Force
}
if (-not $config.local_whisper.compute_type) {
    $config.local_whisper | Add-Member -MemberType NoteProperty -Name "compute_type" -Value "int8" -Force
}
if ($null -eq $config.local_whisper.cpu_threads) {
    $config.local_whisper | Add-Member -MemberType NoteProperty -Name "cpu_threads" -Value 0 -Force
}
$configuredModelPath = [string]$config.voice.vosk_model_path
if ([string]::IsNullOrWhiteSpace($configuredModelPath) -or $configuredModelPath -eq $PreviousDefaultModelPath) {
    Write-Host "Updating config.json to use the larger default Vosk model..."
    $config.voice | Add-Member -MemberType NoteProperty -Name "vosk_model_path" -Value $DefaultModelPath -Force
}
$configJson = $config | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText($ConfigPath, $configJson, [System.Text.UTF8Encoding]::new($false))

if ($SkipVoskModel) {
    Write-Host "Skipping Vosk model download because -SkipVoskModel was provided."
} elseif (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "models") | Out-Null
    if (-not (Test-Path $ModelZip)) {
        Write-Host "Downloading larger default Vosk voice model..."
        Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelZip
    }
    Write-Host "Extracting larger default Vosk voice model..."
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
