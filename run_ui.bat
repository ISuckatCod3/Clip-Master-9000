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
".venv\Scripts\python.exe" -u "control_panel.py" 1>>"logs\ui.out.log" 2>>"logs\ui.err.log"
if errorlevel 1 (
  echo.
  echo UI stopped with an error. See logs\ui.err.log.
  pause
  exit /b 1
)
