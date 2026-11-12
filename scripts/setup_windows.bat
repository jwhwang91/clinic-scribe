@echo off
rem clinic-scribe install for Windows 10/11 (x64, ARM64). Runs scripts\setup.py with Python 3.10-3.13:
rem whisper.cpp official build (SHA-256 checked), .venv inside the repo, models in %LOCALAPPDATA%\clinic-scribe,
rem Start menu and Desktop shortcuts. Messages are printed by setup.py (ASCII only here: cmd.exe code pages).
setlocal
set "SETUP=%~dp0setup.py"
set "PYVER="
for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PYVER (
        py -%%V -c "import sys" >nul 2>nul && set "PYVER=%%V"
    )
)
if defined PYVER (
    py -%PYVER% "%SETUP%" %*
    exit /b
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 (
    python "%SETUP%" %*
    exit /b
)
echo Python 3.10 - 3.13 is required. Install it, then run this again:
echo   winget install Python.Python.3.12
echo   or https://www.python.org/downloads/  (check "Add python.exe to PATH")
exit /b 1
