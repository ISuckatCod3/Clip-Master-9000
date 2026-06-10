@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Run setup.bat first.
  pause
  exit /b 1
)

if not exist "config.json" (
  copy "config.example.json" "config.json" >nul
)

if not exist "logs" mkdir "logs"
".venv\Scripts\python.exe" -u "live_video_interpreter.py" --watch-obs-clips 1>>"logs\obs_renamer.out.log" 2>>"logs\obs_renamer.err.log"
if errorlevel 1 (
  echo.
  echo OBS renamer stopped with an error. See logs\obs_renamer.err.log.
  pause
  exit /b 1
)
