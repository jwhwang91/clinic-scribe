@echo off
rem Open clinic-scribe from a terminal (Windows). Extra arguments go to the app, e.g. scripts\run.bat --open file.wav
setlocal
set "REPO=%~dp0.."
if not exist "%REPO%\.venv\Scripts\python.exe" (
    echo Run scripts\setup_windows.bat first.
    exit /b 1
)
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
"%REPO%\.venv\Scripts\python.exe" -m clinic_scribe %*
