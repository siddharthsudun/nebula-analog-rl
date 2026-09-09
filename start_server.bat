@echo off
REM Double-click (or run from a terminal) to start the silQ dashboard server.
REM Same command as README.md's Quickstart, just Windows-shaped.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

set PYTHONPATH=src

echo Starting server at http://127.0.0.1:8000
echo Press Ctrl+C to stop.
echo.

python -m uvicorn server:app --port 8000

echo.
echo Server stopped.
pause
