@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_whisperlive_server.ps1"
if errorlevel 1 (
  echo.
  echo WhisperLive server failed. See the error above.
  pause
  exit /b 1
)
pause
