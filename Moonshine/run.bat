@echo off
title Moonshine ASR Speech Analyzer

echo ============================================================
echo   Moonshine ASR  ^|  VAD  ^|  Diarization  ^|  Flask
echo ============================================================
echo.

REM ── Use the shared VAD venv which already has all dependencies ───────────────
set VENV_PYTHON=..\VAD-Diarization\.venv\Scripts\python.exe

if not exist %VENV_PYTHON% (
    echo [WARN] VAD venv not found. Creating a new venv...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    set VENV_PYTHON=.venv\Scripts\python.exe
) else (
    echo [OK] Using shared VAD virtual environment.
)

echo.
echo [INFO] HF_TOKEN is configured in .env
echo [INFO] Browser will open automatically at http://localhost:5002
echo [INFO] Moonshine models will start loading automatically
echo [INFO] Moonshine is much smaller than Whisper - loads faster!
echo [INFO] Press Ctrl+C to stop the server
echo.

REM ── Launch Flask app ──────────────────────────────────────────────────────────
%VENV_PYTHON% app.py

pause
