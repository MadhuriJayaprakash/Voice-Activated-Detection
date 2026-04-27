@echo off
title Distil-Whisper Speech Analyzer

echo ============================================================
echo   Distil-Whisper Speech Analyzer
echo   Flask server on http://127.0.0.1:5001
echo ============================================================
echo.

:: Kill any process already using port 5001
echo Checking for existing process on port 5001...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5001 " ^| findstr "LISTENING"') do (
    echo   Stopping old process (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)
echo Done.
echo.

:: Change to the app directory
cd /d "%~dp0"

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Make sure Python is installed and on PATH.
    pause
    exit /b 1
)

:: Check .env file exists
if not exist ".env" (
    echo ERROR: .env file not found.
    echo Create a .env file with:  HF_TOKEN=your_token_here
    pause
    exit /b 1
)

echo Starting Flask app...
echo Open your browser and go to:  http://127.0.0.1:5001
echo.
echo Press Ctrl+C to stop the server.
echo.

python app.py

pause
