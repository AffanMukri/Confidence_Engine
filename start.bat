@echo off
setlocal
cd /d "%~dp0backend"
curl.exe --fail --silent http://127.0.0.1:8765/health >nul 2>nul
if not errorlevel 1 (
    echo Confidence Engine is already running at http://localhost:8765
    echo Open that address in Chrome or Edge.
    pause
    exit /b 0
)
echo Starting Confidence Engine on http://localhost:8765
echo Keep this window open while using the application.
python -m uvicorn main:app --host 127.0.0.1 --port 8765
if errorlevel 1 (
    echo.
    echo The server could not start. Install dependencies with:
    echo   python -m pip install -r requirements.txt
    pause
)
