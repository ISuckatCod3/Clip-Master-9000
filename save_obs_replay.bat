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

".venv\Scripts\python.exe" -u "live_video_interpreter.py" --save-obs-replay-buffer
if errorlevel 1 (
  echo.
  echo Could not save OBS replay.
  pause
  exit /b 1
)
pause
