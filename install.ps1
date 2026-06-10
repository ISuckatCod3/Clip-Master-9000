param(
    [string]$InstallDir = "$env:LOCALAPPDATA\Programs\Clip-Master-9000",
    [string]$Branch = "main",
    [switch]$SkipVoskModel,
    [switch]$NoDesktopShortcut,
    [switch]$LaunchUI
)

$ErrorActionPreference = "Stop"

$RepoZipUrl = "https://github.com/ISuckatCod3/Clip-Master-9000/archive/refs/heads/$Branch.zip"
$TempRoot = Join-Path $env:TEMP ("clip-master-9000-install-" + [guid]::NewGuid().ToString("N"))
$ZipPath = Join-Path $TempRoot "source.zip"
$ExtractPath = Join-Path $TempRoot "source"
$ConfigBackup = $null

function New-Shortcut {
    param(
        [string]$Path,
        [string]$TargetPath,
        [string]$WorkingDirectory,
        [string]$Description
    )

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($Path)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.Description = $Description
    $shortcut.Save()
}

Write-Host "Installing Clip Master 9000"
Write-Host "==========================="
Write-Host ""

New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ExtractPath | Out-Null

try {
    Write-Host "Downloading $RepoZipUrl"
    Invoke-WebRequest -Uri $RepoZipUrl -OutFile $ZipPath

    Write-Host "Extracting source..."
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force
    $SourceDir = Get-ChildItem -Path $ExtractPath -Directory | Select-Object -First 1
    if (-not $SourceDir) {
        throw "Could not find extracted source directory."
    }

    if (Test-Path (Join-Path $InstallDir "config.json")) {
        $ConfigBackup = Join-Path $TempRoot "config.json"
        Copy-Item -Path (Join-Path $InstallDir "config.json") -Destination $ConfigBackup -Force
    }

    Write-Host "Copying files to $InstallDir"
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Copy-Item -Path (Join-Path $SourceDir.FullName "*") -Destination $InstallDir -Recurse -Force

    if ($ConfigBackup) {
        Copy-Item -Path $ConfigBackup -Destination (Join-Path $InstallDir "config.json") -Force
    }

    $buildArgs = @()
    if ($SkipVoskModel) {
        $buildArgs += "-SkipVoskModel"
    }
    if ($LaunchUI) {
        $buildArgs += "-LaunchUI"
    }

    Write-Host "Running local build..."
    & (Join-Path $InstallDir "build.ps1") @buildArgs

    if (-not $NoDesktopShortcut) {
        $Desktop = [Environment]::GetFolderPath("Desktop")
        New-Shortcut `
            -Path (Join-Path $Desktop "Clip Master 9000 UI.lnk") `
            -TargetPath (Join-Path $InstallDir "run_ui.bat") `
            -WorkingDirectory $InstallDir `
            -Description "Open the Clip Master 9000 control panel"
        New-Shortcut `
            -Path (Join-Path $Desktop "Clip Master 9000 Listener.lnk") `
            -TargetPath (Join-Path $InstallDir "run_listener.bat") `
            -WorkingDirectory $InstallDir `
            -Description "Start Clip Master 9000 voice commands"
    }

    Write-Host ""
    Write-Host "Install complete: $InstallDir"
    Write-Host "Run the UI with: $InstallDir\run_ui.bat"
} finally {
    if (Test-Path $TempRoot) {
        Remove-Item -Path $TempRoot -Recurse -Force
    }
}
